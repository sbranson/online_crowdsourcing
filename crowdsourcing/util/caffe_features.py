CAFFE_ROOT = '/home/sbranson/code/other_peoples_code/caffe-master-git/caffe'

import sys
sys.path.insert(0, CAFFE_ROOT + '/python')

import caffe
import numpy as np

GPU = True
CAFFE_ALEXNET = {'model_file':CAFFE_ROOT+'/models/bvlc_reference_caffenet/deploy.prototxt', 'pretrained_file':CAFFE_ROOT+'/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', 'mean':np.load(CAFFE_ROOT+'/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), 'channel_swap':(2,1,0), 'raw_scale':255, 'image_dims':(256,256), 'gpu':GPU, 'layer':'fc6'}
CAFFE_VGG = {'model_file':CAFFE_ROOT+'/models/vgg_ilsvrc_16/deploy.txt', 'pretrained_file':CAFFE_ROOT+'/models/vgg_ilsvrc_16/VGG_ILSVRC_16_layers.caffemodel', 'mean':np.asarray((104,117,123)), 'channel_swap':(2,1,0), 'raw_scale':255.0, 'image_dims':(256,256), 'gpu':GPU, 'layer':'fc6'}

class CaffeFeatureExtractor(object):
  def __init__(self, model_file=None, pretrained_file=None, mean=None, channel_swap=(2,1,0), raw_scale=255, image_dims=(256,256), gpu=True, layer='fc6'):
    if gpu: caffe.set_mode_gpu()
    self.caffe_model = caffe.Classifier(model_file, pretrained_file, mean=mean, channel_swap=channel_swap, raw_scale=raw_scale, image_dims=image_dims)
    self.batch_size = self.caffe_model.blobs[layer].data.shape[0]
    self.feature_dims = self.caffe_model.blobs[layer].data.shape[1]
    self.layer = layer
    
  def extract_features(self, fnames):
    features = np.zeros((len(fnames), self.feature_dims))
    for i in range(0, len(fnames), self.batch_size):
      print str(i)
      input_images = [caffe.io.load_image(fnames[j]) for j in range(i, min(i+self.batch_size,len(fnames)))]
      self.caffe_model.predict(input_images, oversample=False)
      features[i:min(i+self.batch_size,len(fnames)),:] = self.caffe_model.blobs[self.layer].data[:min(self.batch_size,len(fnames)-i),:]
    return features
