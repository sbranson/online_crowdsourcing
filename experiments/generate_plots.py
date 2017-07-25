import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__),"../"))
from crowdsourcing.interfaces.simulator import *
from crowdsourcing.annotation_types.classification import *
from crowdsourcing.annotation_types.part import *
from crowdsourcing.annotation_types.bbox import *
from crowdsourcing.util.caffe_features import *
from crowdsourcing.util.tensorflow_features import *

RAND_PERMS = 3

#bp = BinaryComputerVisionPredictor(CaffeFeatureExtractor(**CAFFE_VGG), num_splits=3)
bp = BinaryComputerVisionPredictor(TensorFlowFeatureExtractor(tfrecords_dir='output/binary/scorpion_tf/tf_records'), num_splits=3, computer_vision_cache='output/binary/scorpion_tf/computer_vision_cache')
full_dataset = CrowdDatasetBinaryClassification(computer_vision_predictor=bp)
full_dataset.load('data/classification/ImageNet_3/scorpion.json')
RunSimulatedExperiments(full_dataset, ALL_METHODS, 'binary/scorpion_tf', ALL_PLOTS, title='Scorpion', num_rand_perms=RAND_PERMS, force_compute=True)

bp = BinaryComputerVisionPredictor(TensorFlowFeatureExtractor(tfrecords_dir='output/binary/beaker_tf/tf_records'), num_splits=3, computer_vision_cache='output/binary/beaker_tf/computer_vision_cache')
full_dataset = CrowdDatasetBinaryClassification(computer_vision_predictor=bp)
full_dataset.load('data/classification/ImageNet_3/beaker.json')
RunSimulatedExperiments(full_dataset, ALL_METHODS, 'binary/beaker', ALL_PLOTS, title='Beaker', num_rand_perms=RAND_PERMS, force_compute=False)

full_dataset = CrowdDatasetParts()
full_dataset.load('data/part/NABirds_1000/mturk.json')
ALL_PLOTS_NO_CV[0]['ylim'] = ALL_PLOTS_NO_CV[1]['ylim'] = ALL_PLOTS_NO_CV[2]['ylim'] = ALL_PLOTS_NO_CV[3]['ylim'] = ALL_PLOTS_NO_CV[4]['ylim'] = ALL_PLOTS_NO_CV[5]['ylim'] = [.05, .1]
RunSimulatedExperiments(full_dataset, ALL_METHODS_NO_CV, 'parts/NABirds_1000', ALL_PLOTS_NO_CV, title='NABirds', num_rand_perms=RAND_PERMS, force_compute=True)

dt = ComputerVisionDetector(num_splits=2)
full_dataset = CrowdDatasetBBox(computer_vision_predictor=dt, debug=2)
full_dataset.load('data/bbox/pedestrians/pedestrians.json')
RunSimulatedExperiments(full_dataset, ALL_METHODS, 'bbox/pedestrians', ALL_PLOTS, title='Caltech Pedestrians', num_rand_perms=RAND_PERMS, force_compute=True)

bp = BinaryComputerVisionPredictor(TensorFlowFeatureExtractor(tfrecords_dir='output/binary/bluebirds_tf/tf_records'), num_splits=3, computer_vision_cache='output/binary/bluebirds_tf/computer_vision_cache')
full_dataset = CrowdDatasetBinaryClassification(computer_vision_predictor=bp)
full_dataset.load('data/classification/bluebirds/bluebirds.json')
RunSimulatedExperiments(full_dataset, ALL_METHODS, 'binary/bluebirds', ALL_PLOTS, title='Welinder Bluebirds', num_rand_perms=10, force_compute=True)
