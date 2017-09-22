import numpy as np
import os
import sys
import math
import urllib

multibox_dir = os.path.join(os.path.dirname(__file__), 'multibox')
sys.path.append(multibox_dir)
from multibox.create_tfrecords import create
from multibox.config import parse_config_file
from multibox.extract import extract_features

IMAGES_PER_SHARD = 10000
NUM_THREADS = 5
MODEL_FILE_URL = 'http://vision.caltech.edu/~sbranson/online_crowdsourcing/multibox_models'


class TensorFlowFeatureExtractor(object):
  def __init__(self, checkpoint_path=None, config_file=None, layer='PreLogits', tfrecords_dir='tf_tmp', batch_size=None, model_name=None, dataset_name = 'dataset'): #layer='Mixed_7c'
    self.checkpoint_path = checkpoint_path
    self.layer = layer
    self.tfrecords_dir = tfrecords_dir
    self.dataset_name = dataset_name
    self.cfg = parse_config_file(config_file if config_file else os.path.join(multibox_dir, 'config', 'config_classify.yaml'))
    if batch_size != None:
      self.cfg.BATCH_SIZE = batch_size
    if model_name != None:
      self.cfg.MODEL_NAME = model_name
    
    if self.checkpoint_path is None:
      self.checkpoint_path = os.path.join(multibox_dir, 'models', 'inception_v3.ckpt')
      if not os.path.exists(self.checkpoint_path):
        urllib.urlretrieve(MODEL_FILE_URL+'/inception_v3.ckpt', self.checkpoint_path)

  
  def format_dataset(self, fnames):
    dataset = [{"filename":fnames[i], "id":i, "class":{"label":0,"text":"none","conf":1}} for i in range(len(fnames))]
    return dataset
  
  def create_tf_records(self, dataset, nshards):  
    # Create tf records files for each image
    if not os.path.exists(self.tfrecords_dir): 
      os.makedirs(self.tfrecords_dir)
    num_threads = min(nshards, NUM_THREADS)
    failed_images = create(
      dataset=dataset,
      dataset_name=self.dataset_name,
      output_directory=self.tfrecords_dir,
      num_shards=nshards,
      num_threads=num_threads,
      shuffle=False)
    
    # Print failed image ids, and create a lookup array to handle tf record files
    # that omit failed images.  If i is an index into saved tf records, good_images[i]
    # is the index into the corresponding image in dataset
    failed_ids = {}
    if len(failed_images) > 0:
      print("%d images failed." % (len(failed_images),))
      for image_data in failed_images:
        failed_ids[image_data['id']] = image_data
        print("Image %s: %s" % (image_data['id'], image_data['error_msg']))
    good_images = []
    for i in range(len(dataset)):
      if not dataset[i]['id'] in failed_ids:
        good_images.append(i)
    
    return good_images

  def extract_features(self, fnames):
    print "Extracting Tensorflow features " + self.checkpoint_path + " for " + str(len(fnames)) + " images..."
    dataset = self.format_dataset(fnames)
    nshards = int(math.ceil(float(len(dataset))/IMAGES_PER_SHARD))
    good_images = self.create_tf_records(dataset, nshards)
    
    tfrecords = [os.path.join(self.tfrecords_dir, f) for f in os.listdir(self.tfrecords_dir) if os.path.isfile(os.path.join(self.tfrecords_dir, f))]
    dfeats = extract_features(tfrecords, self.checkpoint_path, int(math.ceil(len(dataset)/float(self.cfg.BATCH_SIZE))), [self.layer], self.cfg)
    feats = dfeats[self.layer]
    ids = [int(i) for i in dfeats['ids']]
    features = np.zeros((len(dataset), feats.shape[1]))
    features[ids,:] = feats
    return features
