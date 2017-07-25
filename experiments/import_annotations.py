import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__),"../"))
from crowdsourcing.util.data_convert import *

BIRD_PARTS = ['bill', 'crown', 'nape', 'left eye', 'right eye', 'belly', 'breast', 'back', 'tail', 'left wing', 'right wing']

import_dataset_cubam('data/classification/bluebirds/bluebirds_labels.yaml', 'data/classification/bluebirds/bluebirds.json', gt_yaml_file_in='data/classification/bluebirds/bluebirds_gt.yaml', base_url="http://sbranson.no-ip.org/bluebirds", image_dir='data/classification/bluebirds/images')

import_dataset_vibe('data/classification/cub_40/class_labels_to_annotations.json', 'data/classification/cub_40', image_dir='data/classification/cub_40/images')

import_bbox_dataset_old_server('data/bbox/pedestrians/179_wh.json', 'data/bbox/pedestrians/pedestrians.json', expert_json_file_in='data/bbox/pedestrians/177(3)_wh.json', image_dir='data/bbox/pedestrians/images')

import_part_dataset_old_server('data/part/NABirds_1000/134.json', 'data/part/NABirds_1000/mturk.json', expert_json_file_in='data/part/NABirds_1000/141.json', part_names=BIRD_PARTS, image_dir='data/part/NABirds_1000/images')
import_part_dataset_old_server('data/part/NABirds_1000/142.json', 'data/part/NABirds_1000/cturk.json', expert_json_file_in='data/part/NABirds_1000/141.json', part_names=BIRD_PARTS, image_dir='data/part/NABirds_1000/images')
