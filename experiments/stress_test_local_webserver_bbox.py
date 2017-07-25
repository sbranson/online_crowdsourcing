import threading
import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__),"../"))
from crowdsourcing.annotation_types.bbox import *
from crowdsourcing.interfaces.local_webserver import *

DATASET_DIR = "data/bbox/pedestrians"
ASSIGNMENT_DATA = os.path.join(DATASET_DIR, "pedestrians.json")

IMAGE_DIR = 'data/bbox/pedestrians/images'

OUTPUT_FOLDER = 'pedestrians_stress_test'
ONLINE = True
WORKERS_PER_IMAGE = 0
HOST = 'sbranson.no-ip.org'
PARAMS = {'instructions':'Draw a box around each pedestrian in the image', 'example_url':'', 'object_name':'pedestrian'}

full_dataset = CrowdDatasetBBox(debug=2, learn_worker_params=True, learn_image_params=True)
full_dataset.load(ASSIGNMENT_DATA)

dataset = CrowdDatasetBBox(name='pedestrians')
dataset.scan_image_directory(IMAGE_DIR)
crowdsource = LocalCrowdsourcer(dataset, HOST, OUTPUT_FOLDER, max_annos=7, hit_params = PARAMS, online = ONLINE, thumbnail_size = (100,100), initial_assignments_per_image=WORKERS_PER_IMAGE, port=8080)

threading.Thread(target=crowdsource.run).start()

while not hasattr(crowdsource, 'hits') or not crowdsource.hits:
  time.sleep(1) 
StressTestLocalCrowdsourcer(HOST+':8080', full_dataset)

