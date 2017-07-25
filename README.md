# Online Crowdsourcing 
This is an implementation of the paper 
```
  Lean Crowdsourcing: Combining Humans and Machines in an Online System ([pdf](http://vision.caltech.edu/~sbranson/online_crowdsourcing/online_crowdsourcing_cvpr2017.pdf)) 
  Steve Branson, Grant Van Horn, Pietro Perona
  CVPR 2017
```

It contains code for annotating images with class, bounding box, and part labels, interfacing with Amazon Mechanical Turk, combining worker labels while modeling worker skill, training up computer vision classifiers and detectors interactively, and combined human/computer prediction.

# Collecting Different Types of Annotations

## Collecting Classification Datasets Like ImageNet
This is a simple example to collect datasets for binary or multiclass classification, similar to ImageNet, CUB200, or Caltech256.  See experiments/collect_annotations_imagenet3.py for an example.  

You will need an AWS ACCESS KEY for paying for Amazon Mechanical Turk services.  See http://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSGettingStartedGuide/AWSCredentials.htm and set AWS_ACCESS_KEY and AWS_SECRET_ACCESS_KEY in keys.json.  

Suppose you want to collect a dataset of images of object classes, such as scorpion, beaker, beagle, etc.  A common approach is to download candidate images using an image search engine, then using Mechanical Turk to filter the results.  If you want to automatically download images from Flickr, see https://www.flickr.com/services/api/misc.api_keys.html and set FLICKR_API_KEY and FLICKR_API_SECRET_KEY in keys.json.  The following code will download Flickr images of scorpions into the folder output/scorpion/flickr:
```
  import json
  import os
  from crowdsourcing.util.image_search import *
  with open('keys.json') as f: keys = json.load(f)
  image_folder = os.path.join('output', 'scorpion', 'flickr')
  FlickrImageSearch('scorpion', image_folder, keys.FLICKR_API_KEY, FLICKR_API_SECRET_KEY, max_photos=MAX_PHOTOS)
```
You can obtain annotated results using
```
  from crowdsourcing.interfaces.mechanical_turk import *
  from crowdsourcing.annotation_types.classification import *
  from crowdsourcing.util.tensorflow_features import *
  INSTRUCTIONS = { 'object_name' : 'scorpion', 'definition' : 'Arachnid of warm dry regions having a long segmented tail ending in a venomous stinger', 'search' : ['scorpion', 'scorpion arachnid'], 'wikipedia_url' : 'https://en.wikipedia.org/wiki/Scorpion', 'example_image_urls' : ['http://imagenet.stanford.edu/nodes/2/01770393/b0/b02dcf2c1d8c7a735b52ab74300c342124e4be5c.thumb', 'http://imagenet.stanford.edu/nodes/2/01770393/31/31af6ea97dd040ec2ddd6ae86fe1f601ecfc8c02.thumb', 'http://imagenet.stanford.edu/nodes/2/01770393/38/382e998365d5667fc333a7c8f5f6e74e3c1fe164.thumb', 'http://imagenet.stanford.edu/nodes/2/01770393/88/88bc0f14c9779fad2bc364f5f4d8269d452e26c2.thumb'] },
  output_folder = os.path.join('output', 'scorpion')
  bp = BinaryComputerVisionPredictor(TensorFlowFeatureExtractor(tfrecords_dir=output_folder+'/tf_records'), num_splits=3, computer_vision_cache=output_folder+'/computer_vision_cache')
  dataset = CrowdDatasetBinaryClassification(name='scorpion', learn_worker_params=True, learn_image_params=False, computer_vision_predictor=bp)
  dataset.scan_image_directory(os.path.join(image_folder, 'images'))
  crowdsource = MTurkCrowdsourcer(dataset, keys.AWS_ACCESS_KEY, keys.AWS_SECRET_ACCESS_KEY, HOST, output_folder, sandbox=False, hit_params = INSTRUCTIONS, thumbnail_size = (100,100), online=True) 
  crowdsource.run()
```
where HOST is the host name of your computer (it will be used to host a web server for mturk tasks).  The variable INSTRUCTIONS contains optional info to present in the MTurk annotation GUI.  The parameters learn_worker_params and learn_image_params enable modeling of worker skill and image difficulty in the model.  The parameter online enables online crowdsourcing (using a variable number of workers per image.  One can disable computer vision by omitting the computer_vision_predictor parameter.

## Collecting Bounding Box Annotations

This is a simple example to collect datasets for annotating bounding boxes around objects in an image, similar to VOC experiments/collect_annotations_bbox.py for an example.

You will need an AWS ACCESS KEY for paying for Amazon Mechanical Turk services.  See http://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSGettingStartedGuide/AWSCredentials.htm and set AWS_ACCESS_KEY and AWS_SECRET_ACCESS_KEY in keys.json.

The following example obtains bounding boxes around pedestrians in a dataset of images:
```
  from crowdsourcing.interfaces.mechanical_turk import *
  from crowdsourcing.annotation_types.bbox import *
  with open('keys.json') as f: keys = json.load(f)
  PARAMS = {'instructions':'Draw a box around each pedestrian in the image', 'example_url':'', 'object_name':'pedestrian'}
  dataset = CrowdDatasetBBox(name='pedestrians')
  dataset.scan_image_directory(IMAGE_DIR)
  crowdsource = MTurkCrowdsourcer(dataset, keys.AWS_ACCESS_KEY, keys.AWS_SECRET_ACCESS_KEY, HOST, OUTPUT_FOLDER, sandbox=FALSE,
                                  hit_params = PARAMS, online = True, thumbnail_size = (100,100))
```

# Generating Results From The CVPR paper

See the file experiments/generate_plots.py to generate results from the CVPR 2017 paper.  In this case, we first obtained an excessive amount of mturk annotations for each image, then simulated results where a subset of annotations are used.  The following code will run experiments for binary classification, bounding box annotation, and part annotation, comparing a wide variety of baselines and lesioned versions of the model:
```
  import sys
  import os
  from crowdsourcing.interfaces.simulator import *
  from crowdsourcing.annotation_types.classification import *
  from crowdsourcing.annotation_types.part import *
  from crowdsourcing.annotation_types.bbox import *
  from crowdsourcing.util.caffe_features import *
  from crowdsourcing.util.tensorflow_features import *
  
  RAND_PERMS = 3
  bp = BinaryComputerVisionPredictor(TensorFlowFeatureExtractor(tfrecords_dir='output/binary/scorpion_tf/tf_records'), num_splits=3, computer_vision_cache='output/binary/scorpion_tf/computer_vision_cache')
  full_dataset = CrowdDatasetBinaryClassification(computer_vision_predictor=bp)
  full_dataset.load('data/classification/ImageNet_3/scorpion.json')
  RunSimulatedExperiments(full_dataset, ALL_METHODS, 'binary/scorpion_tf', ALL_PLOTS, title='Scorpion', num_rand_perms=RAND_PERMS, force_compute=True)
  
  full_dataset = CrowdDatasetParts()
  full_dataset.load('data/part/NABirds_1000/mturk.json')
  ALL_PLOTS_NO_CV[0]['ylim'] = ALL_PLOTS_NO_CV[1]['ylim'] = ALL_PLOTS_NO_CV[2]['ylim'] = ALL_PLOTS_NO_CV[3]['ylim'] = ALL_PLOTS_NO_CV[4]['ylim'] = ALL_PLOTS_NO_CV[5]['ylim'] = [.05, .1]
  RunSimulatedExperiments(full_dataset, ALL_METHODS_NO_CV, 'parts/NABirds_1000', ALL_PLOTS_NO_CV, title='NABirds', num_rand_perms=RAND_PERMS, force_compute=True)
  
  dt = ComputerVisionDetector(num_splits=2)
  full_dataset = CrowdDatasetBBox(computer_vision_predictor=dt, debug=2)
  full_dataset.load('data/bbox/pedestrians/pedestrians.json')
RunSimulatedExperiments(full_dataset, ALL_METHODS, 'bbox/pedestrians', ALL_PLOTS, title='Caltech Pedestrians', num_rand_perms=RAND_PERMS, force_compute=True)
```
  

