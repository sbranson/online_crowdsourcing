import cPickle as pickle
import glob
import json
import math
import numpy as np
import os
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
import sys
import math
import urllib

from bbox import CrowdLabelBBox


multibox_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'util', 'multibox')
sys.path.append(multibox_dir)

from ...util.multibox.config import *
from ...util.multibox.train import *
from ...util.multibox.detect import *
from ...util.multibox.create_tfrecords import *
#from ...util.multibox.eval import *

MODEL_FILE_URL = 'http://vision.caltech.edu/~sbranson/online_crowdsourcing/multibox_models'

class ComputerVisionDetector(object):
  def __init__(self, num_splits=2, 
    bbox_priors_file=None, 
    train_cfg_file=None, 
    detect_cfg_file=None, 
    images_dir=None, 
    train_dir=None, 
    initial_checkpoint_path=None, 
    num_epochs_per_train_step=4, 
    max_detections_per_image=20):

    self.iteration = 0
    self.num_splits = 2
    self.images_dir = images_dir
    self.train_dir = train_dir
    self.max_detections_per_image = max_detections_per_image
    self.num_epochs_per_train_step = num_epochs_per_train_step
    self.bbox_priors_file = bbox_priors_file
    self.train_cfg_file = train_cfg_file
    self.detect_cfg_file = detect_cfg_file
    self.initial_checkpoint_path = initial_checkpoint_path
    
    if self.bbox_priors_file is None:
      self.bbox_priors_file = os.path.join(multibox_dir, 'models', 'coco_person_priors_7_warped_non_restricted.pkl')
      if not os.path.exists( self.bbox_priors_file):
        urllib.urlretrieve(MODEL_FILE_URL+'/coco_person_priors_7_warped_non_restricted.pkl', self.bbox_priors_file)
    if self.initial_checkpoint_path is None:
      self.initial_checkpoint_path = os.path.join(multibox_dir, 'models', 'model.ckpt-141123')
      if not os.path.exists( self.initial_checkpoint_path):
        urllib.urlretrieve(MODEL_FILE_URL+'/model.ckpt-141123', self.initial_checkpoint_path)
        urllib.urlretrieve(MODEL_FILE_URL+'/model.ckpt-141123.meta', self.initial_checkpoint_path+'.meta')
    if self.train_cfg_file is None:
      self.train_cfg_file = os.path.join(multibox_dir, 'config', 'config_train.yaml')
    if self.detect_cfg_file is None:
      self.detect_cfg_file = os.path.join(multibox_dir, 'config', 'config_detect.yaml')
    
  def predict_probs(self, images, labels, valid_train=None, cache_name=None, cv_worker=None, naive=True):
    """
    Args:
        images: an array of CrowdImageBBox
        labels: an array of CrowdLabelBBox
    """

    print
    print
    print "###################################"
    print "Predict Probs %d: %d valid images" % (self.iteration, sum(valid_train))

    # On iteration 0, just return an empty label for every image
    if self.iteration == 0 or sum(valid_train) == 0:
      cv_crowd_labels = []
      for image in images:
        cv_pred = CrowdLabelBBox(image, cv_worker)
        cv_pred.bboxes = []
        cv_crowd_labels.append(cv_pred)
     
    else:
      #Grab the "valid" images that we can use for training
      valid_image_ids = []
      for i, image in enumerate(images):
        if valid_train[i]:
          valid_image_ids.append(image.id)
      
      cv_crowd_labels = self.do_step(images, valid_image_ids, cv_worker, naive)
      
      # Save off the results of this iteration
      logdir = os.path.join(self.train_dir, "%d" % (self.iteration,))
      if not os.path.exists(logdir):
        os.makedirs(logdir)
      with open(os.path.join(logdir, "cv_crowd_labels.pkl"), 'w') as f:
        pickle.dump(cv_crowd_labels, f)
    
    self.iteration += 1

    return cv_crowd_labels

  def do_step(self, images, valid_image_ids, cv_worker, naive=False):
    """
    Args:
      images (list): a list of "valid" images to be used in this step
      valid_image_ids (list): a list indicating which images can be used for training
    """

    assert self.iteration > 0

    # Convert steve's data to the multibox format
    encoded_dataset = {}
    encoded_dataset['images'] = {}
    for image in images:
      encoded_dataset['images'][image.id] = image.encode()
    
    encoded_dataset['combined_labels'] = []
    for image in images:
      if hasattr(image,'y') and image.y:
        encoded_dataset['combined_labels'].append({'image_id': image.id, 'label': image.y.encode()})
    
    dataset = convert_crowdsourcing_to_tf_format(encoded_dataset['images'], encoded_dataset['combined_labels'], self.images_dir)

    cv_crowd_labels = []

    # Create the different train/test splits 
    dataset = np.array(dataset)
    kf = KFold(n_splits=self.num_splits, shuffle=True)
    for split, (train_split, test_split) in enumerate(kf.split(dataset)):

      train_dataset = dataset[train_split].tolist()
      test_dataset = dataset[test_split].tolist()

      # Filter the train dataset to include only images that have labels
      train_dataset = [image for image in train_dataset if image['id'] in valid_image_ids]
      num_train_images = len(train_dataset)

      # Create a val dataset that only contains test images that have labels
      val_dataset = [image for image in test_dataset if image['id'] in valid_image_ids]
      num_val_images = len(val_dataset)

      num_test_images = len(test_dataset)

      if num_train_images > 0 and num_val_images > 0:
        
        # Figure out the maximum number of boxes in one of the images
        max_num_bboxes_train = max([image['object']['bbox']['count'] for image in train_dataset])
        max_num_bboxes_val = max([image['object']['bbox']['count'] for image in val_dataset])
        max_num_bboxes_test = max([image['object']['bbox']['count'] for image in test_dataset])

        print "Train Dataset Stats: %d images, %d max bboxes" % (num_train_images, max_num_bboxes_train)
        print "Val Dataset Stats: %d images, %d max bboxes" % (num_val_images, max_num_bboxes_val)
        print "Test Dataset Stats: %d images, %d max bboxes" % (num_test_images, max_num_bboxes_test)

        # Create the logdir for this step
        logdir = os.path.join(self.train_dir, "%d-%d" % (self.iteration,split))
        
        # Create the tfrecords
        tfrecords_dir = os.path.join(logdir, 'tfrecords')
        if not os.path.exists(tfrecords_dir):
          os.makedirs(tfrecords_dir) 
        train_tfrecords = self.create_tfrecords(train_dataset, tfrecords_dir, 'train')
        test_tfrecords = self.create_tfrecords(test_dataset, tfrecords_dir, 'test')
        val_tfrecords = self.create_tfrecords(val_dataset, tfrecords_dir, 'val')

        # Train, Eval, Detect
        self.do_train_step(num_train_images, train_tfrecords, self.initial_checkpoint_path, logdir, max_num_bboxes_train)
        checkpoint_path = logdir
        #self.do_eval_step(num_val_images, val_tfrecords, logdir, checkpoint_path, max_num_bboxes_val)
        results = self.do_detect_step(num_test_images, test_tfrecords, logdir, checkpoint_path, max_num_bboxes_test)

        # Convert the detections to encoded crowd labels
        image_annotations = convert_tf_to_crowdsourcing_format(encoded_dataset['images'], results)
        # Decode the crowd labels
        crowd_labels = self.convert_detections_to_crowd_labels(images, image_annotations, cv_worker)

        # build the gt crowd labels
        gt_image_ids = set([image['id'] for image in val_dataset])
        gt_images = [image for image in images if image.id in gt_image_ids]
        gt_labels = [image.y for image in gt_images]

        
        if naive:
          self.threshold_crowd_labels(crowd_labels, gt_labels)
        else:
          # Calibrate the crowd labels and set the prob_fp
          self.calibrate_computer_vision(crowd_labels, gt_labels)
      
      else:
        # Just add empty bounding boxes?
        test_image_ids = set([image['id'] for image in test_dataset]) 
        crowd_labels = []
        for image in images:
          if image.id in test_image_ids:
            cv_pred = CrowdLabelBBox(image, cv_worker)
            cv_pred.bboxes = []
            crowd_labels.append(cv_pred)
      
      cv_crowd_labels += crowd_labels

    return cv_crowd_labels

  def convert_detections_to_crowd_labels(self, images, encoded_labels, cv_worker):
    """
    Args:
      images (list): a list of CrowdImageBBox
      encoded_labels (list): an array of encoded crowd labels
    Returns:
      list : a list of CrowdLabelBBox
    """

    crowd_labels = []
    for image in images:
      if image.id in encoded_labels:
        cv_pred = CrowdLabelBBox(image, cv_worker)
        cv_pred.parse(encoded_labels[image.id]['anno'])
        crowd_labels.append(cv_pred)
    return crowd_labels
  
  def create_tfrecords(self, dataset, save_dir, prefix):
    """
    Args:
      dataset (list): a dataset ready to be converted to tfrecords
      save_dir (str): directory location where to store the tfrecords
      prefix (str): a prefix for the tfrecord file names
    Returns:
      list : a list of the created tfrecord file paths
    """
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    
    create_tfrecords.create(
      dataset,
      prefix,
      save_dir,
      num_shards=1,
      num_threads=1
    )
    tfrecords = glob.glob(os.path.join(save_dir, prefix+'*'))

    return tfrecords
  
  def do_train_step(self, num_images, tfrecords, pretrained_model_path, logdir, max_num_bboxes):

    # Make sure the logdir exists
    if not os.path.exists(logdir):
      os.makedirs(logdir)
    
    cfg = parse_config_file(self.train_cfg_file)
    cfg.BATCH_SIZE = min(num_images, cfg.BATCH_SIZE)
    cfg.NUM_TRAIN_ITERATIONS = np.ceil(num_images * float(self.num_epochs_per_train_step) / cfg.BATCH_SIZE)
    cfg.QUEUE_MIN = min(num_images, 50)
    cfg.MAX_NUM_BBOXES = max_num_bboxes

    with open(self.bbox_priors_file) as f:
      bbox_priors = pickle.load(f)
    bbox_priors = np.array(bbox_priors).astype(np.float32)

    train.train(
      tfrecords=tfrecords,
      bbox_priors=bbox_priors,
      logdir=logdir,
      cfg=cfg,
      pretrained_model_path=pretrained_model_path,
      fine_tune = False,
      trainable_scopes = None,
      use_moving_averages = True,
      restore_moving_averages = True
    )
  
  def do_eval_step(self, num_images, tfrecords, logdir, checkpoint_path, max_num_bboxes):

    eval_summaries_dir = os.path.join(logdir, "eval_summaries")
    if not os.path.exists(eval_summaries_dir):
      os.makedirs(eval_summaries_dir)
    
    # Choose a batch size that will ensure that we eval on all images.
    num_images_factors = factors(num_images)
    batch_size = 1
    for f in num_images_factors:
      if f > batch_size and f <= 32:
        batch_size = f

    cfg = parse_config_file(self.detect_cfg_file)
    cfg.BATCH_SIZE = batch_size
    cfg.MAX_NUM_BBOXES = max_num_bboxes

    with open(self.bbox_priors_file) as f:
      bbox_priors = pickle.load(f)
    bbox_priors = np.array(bbox_priors).astype(np.float32)
    
    eval.eval(
      tfrecords=tfrecords,
      bbox_priors=bbox_priors,
      summary_dir=eval_summaries_dir,
      checkpoint_path=checkpoint_path,
      max_iterations = 0,
      cfg=cfg
    )

  def do_detect_step(self, num_images, tfrecords, logdir, checkpoint_path, max_num_bboxes):

    # Create the location to save the results
    save_dir = os.path.join(logdir, "results")
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    
    # Choose a batch size that will ensure that we test on all images.
    num_images_factors = factors(num_images)
    batch_size = 1
    for f in num_images_factors:
      if f > batch_size and f <= 32:
        batch_size = f

    cfg = parse_config_file(self.detect_cfg_file)
    cfg.BATCH_SIZE = batch_size
    cfg.MAX_NUM_BBOXES = max_num_bboxes

    with open(self.bbox_priors_file) as f:
      bbox_priors = pickle.load(f)
    bbox_priors = np.array(bbox_priors).astype(np.float32)
    
    detect.detect(
      tfrecords=tfrecords,
      bbox_priors=bbox_priors,
      checkpoint_path=checkpoint_path,
      save_dir = save_dir,
      max_detections = self.max_detections_per_image,
      max_iterations = 0,
      cfg=cfg
    )

    # Get the results
    result_path = max(glob.iglob(os.path.join(save_dir, "*")), key=os.path.getctime)
    with open(result_path) as f:
      results = json.load(f)
    
    return results 
  
  # Run this on the validation set
  # This function matches detections to ground truth bounding boxes, then trains a probabilistic classifier
  # going from detection score to detection probability
  def calibrate_computer_vision(self, cv_preds, gt_labels):
    """
    Args:
      cv_preds (list):  a list of CrowdLabelBBox
      gt_labels (list): a list of CrowdLabelBBox. This can be a subset of the images found in cv_preds
    """
    
    # cv_preds could contain images that don't have any ground truth labels. 
    # we need to make sure we filter those out.
    val_image_ids = set([label.image.id for label in gt_labels])

    val_detection_scores, val_is_true_positive = [], []

    for i in range(len(cv_preds)):
      
      # We only want val images
      if cv_preds[i].image.id in val_image_ids:

        # Computer assignments between detection and ground truth
        for b in cv_preds[i].bboxes: 
          b.prob_fp = 1-b.score # Use computer vision estimated as an approximate false positive probability
          b.sigma = cv_preds[i].worker.params.prior_sigma_prior  # global prior
        cv_preds[i].match_to(gt_labels[i], match_by='prob')
        
        val_detection_scores += [math.log(b.score) for b in cv_preds[i].bboxes]  # detection score (resembling log probability)
        val_is_true_positive += [float(not b.a is None) for b in cv_preds[i].bboxes]
    
    val_detection_scores = np.array(val_detection_scores).reshape(-1, 1)

    # Now gather all of the detection scores:
    test_detection_scores = []
    for i in range(len(cv_preds)):
      for b in cv_preds[i].bboxes:
        b.sigma = cv_preds[i].worker.params.prior_sigma_prior  # global prior
        test_detection_scores.append(math.log(b.score))
    
    test_detection_scores = np.array(test_detection_scores).reshape(-1, 1)
    
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(val_detection_scores, val_is_true_positive)
    clf_prob = CalibratedClassifierCV(clf, cv='prefit', method='sigmoid')
    clf_prob.fit(val_detection_scores, val_is_true_positive)
    p = clf_prob.predict_proba(test_detection_scores)
    j = 0
    for i in range(len(cv_preds)):
      for k in range(len(cv_preds[i].bboxes)):
        cv_preds[i].bboxes[k].prob_fp = float(p[j+k][0])
      j += len(cv_preds[i].bboxes)

  def threshold_crowd_labels(self, cv_preds, gt_labels):
    """
    Args:
      cv_preds (list):  a list of CrowdLabelBBox
      gt_labels (list): a list of CrowdLabelBBox. This can be a subset of the images found in cv_preds
    """
    
    # Calibrate the detections, then threshold on prob_fp >= 0.5
    if True:
      self.calibrate_computer_vision(cv_preds, gt_labels)

      # Threshold the detections
      for i in range(len(cv_preds)):
        cv_preds[i].bboxes = [bbox for bbox in cv_preds[i].bboxes if bbox.prob_fp < 0.5]
      
      return
    
    # Sweep the scores, looking for the sweet spot
    else:
    

      # cv_preds could contain images that don't have any ground truth labels. 
      # we need to make sure we filter those out.
      val_image_ids = set([label.image.id for label in gt_labels])
      val_image_dict = {label.image.id : label for label in gt_labels}

      if False:
        # Get the scores from the detections
        val_image_bbox_scores = []
        for i in range(len(cv_preds)):
          # We only want val images
          if cv_preds[i].image.id in val_image_ids:
            for b in cv_preds[i].bboxes:
              val_image_bbox_scores.append(b.score)  

        val_image_bbox_scores.sort()
        val_image_bbox_scores.reverse()

        # We probably don't need to cycle through all of the scores
        # Is the loss convex? Should we just check for it to increase?
        val_image_bbox_scores = val_image_bbox_scores[:len(gt_labels) * 4]
      
      else:
        # Just used some fixed thresholds
        val_image_bbox_scores = np.arange(1, 0, -.05)

      #print "Checking %d scores for the best threshold" % (len(val_image_bbox_scores),)

      minimum_loss = np.inf
      best_threshold = -np.inf
      for threshold_score in val_image_bbox_scores:
        total_loss = 0.
        for i in range(len(cv_preds)):
          cv_pred = cv_preds[i]
          # We only want val images
          if cv_pred.image.id in val_image_ids:
            orig_bboxes = cv_pred.bboxes
            cv_pred.bboxes = []
            for b in orig_bboxes:
              if b.score >= threshold_score:
                cv_pred.bboxes.append(b)
            total_loss += cv_pred.loss(val_image_dict[cv_pred.image.id])
            cv_pred.bboxes = orig_bboxes
        
       # print "Threshold: %0.5f \t Total Loss %0.3f vs Best Loss %0.3f" % (threshold_score, total_loss, minimum_loss)

        if total_loss < minimum_loss:
          best_threshold = threshold_score
          minimum_loss = total_loss

      #print "Minimum Loss: %0.3f" % (minimum_loss,)
      #print "Best Threshold: %0.3f" % (best_threshold,)

      # Threshold the detections
      for i in range(len(cv_preds)):
        cv_preds[i].bboxes = [bbox for bbox in cv_preds[i].bboxes if bbox.score >= best_threshold]



# Compute the factors of a number
def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def convert_crowdsourcing_to_tf_format(images, annotations, image_dir, remove_images_without_instances=False):
    """
    Args:
        images: dictionary {image_id : image info} (images from Steve's code)
        annotations: list of annotation dictionaries (combined_labels from Steve's code)
        image_dir: directory where images reside
    """

    annotation_data = []
    gt_annotation_id = 1
    found_image_ids = set()

    dataset = {}
    for image_id, image_info in images.iteritems():
        
        image_path = os.path.join(image_dir, image_info['fname'])
        dataset[image_id] = { 
            "filename" : image_path,
            "id" : str(image_id),
            "height" : image_info['height'],
            "width" : image_info['width'],
            "class" : {
                "label" : 1,
                "text" : "person",
            },
            "object" : { 
                "bbox" : {
                    "xmin" : [],
                    "xmax" : [],
                    "ymin" : [],
                    "ymax" : [],
                    "label" : [],
                    "count" : 0
                },
                "parts" : {
                    "x" : [],
                    "y" : [],
                    "v" : []
                },
                "area" : [], # segmentation area
                "id" : [] # annotation id
            }
        }
    
    annotation_id = 0
    for anno_data in annotations:

        image_id = anno_data['image_id']
        object_data = dataset[image_id]["object"]
        
        image_width = float(dataset[image_id]['width'])
        image_height = float(dataset[image_id]['height'])

        anno = anno_data['label']
        bboxes = anno['bboxes']
        for bbox in bboxes:

            #image_width = float(bbox['image_width'])
            #image_height = float(bbox['image_height'])

            x1 = bbox['x']
            y1 = bbox['y']
            x2 = bbox['x2']
            y2 = bbox['y2']

            if x2 < x1:
                t = x1
                x1 = x2
                x2 = t
            
            if y2 < y1:
                t = y1
                y1 = y2
                y2 = t

            area = (x2 - x1) * (y2 - y1)

            xmin = np.clip(x1 / image_width, 0., 1.)
            ymin = np.clip(y1 / image_height, 0., 1.)
            xmax = np.clip(x2 / image_width, 0., 1.)
            ymax = np.clip(y2 / image_height, 0., 1.)

            object_data['bbox']['xmin'].append(xmin)
            object_data['bbox']['ymin'].append(ymin)
            object_data['bbox']['xmax'].append(xmax)
            object_data['bbox']['ymax'].append(ymax)
            object_data['bbox']['label'].append(1)
            object_data['bbox']['count'] += 1
            object_data['area'].append(area)
            object_data['id'].append(annotation_id)
            annotation_id += 1

    # remove images with no people?
    if remove_images_without_instances: 
      final_dataset = []
      max_num_bboxes = 0
      for image in dataset.values():
          if image['object']['bbox']['count'] > 0:
              final_dataset.append(image)
              max_num_bboxes = max(max_num_bboxes, image['object']['bbox']['count'])
      print "Max Number of BBoxes: %d" % (max_num_bboxes,)
      return final_dataset
    else:
      return dataset.values()

def convert_tf_to_crowdsourcing_format(images, detections):
    """
    Args:
        images: dictionary {image_id : image info} (images from Steve's code)
        detections: detection output from multibox
    Returns:
      dict : a dictionary mapping image_ids to bounding box annotations
    """
    image_annotations = {}
    for detection in detections:
        image_id = str(detection['image_id'])
        score = detection['score']
        x1, y1, x2, y2 = detection['bbox']

        # GVH: Do we want to check for reversal or area issues? 

        image_width = images[image_id]['width']
        image_height = images[image_id]['height']

        if image_id not in image_annotations:
            image_annotations[image_id] = {
                "anno" : {
                    "bboxes" : []
                },
                "image_id": image_id, 
                "image_width": image_width, 
                "image_height": image_height
            }

        bboxes = image_annotations[image_id]["anno"]["bboxes"]

        bboxes.append({
            "x": x1 * image_width, 
            "y": y1 * image_height, 
            "x2": x2 * image_width,
            "y2": y2 * image_height,
            "image_height": image_height, 
            "image_width": image_width,
            "score" : score
        })
    
    return image_annotations



