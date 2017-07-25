import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#sys.path.append(os.path.join(os.path.dirname(__file__),"../"))
from crowdsourcing.interfaces.mechanical_turk import *
from crowdsourcing.interfaces.local_webserver import *
from crowdsourcing.util.image_search import *
from crowdsourcing.annotation_types.classification import *
from crowdsourcing.annotation_types.bbox import *
from crowdsourcing.annotation_types.part import *

# directory containing the images we want to annotate
IMAGE_DIR = 'data/bbox/pedestrians_small/images'

OUTPUT_FOLDER = 'pedestrians_small2'

USE_MTURK = False#True
ONLINE = True
WORKERS_PER_IMAGE = 0

# Amazon account information for paying for mturk tasks, see 
# http://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSGettingStartedGuide/AWSCredentials.htm
AWS_ACCESS_KEY = 'AKIAJKKZFJJQTDBSF4GA'
AWS_SECRET_ACCESS_KEY = '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'
SANDBOX = False

HOST = 'sbranson.no-ip.org'

PARAMS = {'instructions':'Draw a box around each pedestrian in the image', 'example_url':'', 'object_name':'pedestrian'}

dataset = CrowdDatasetBBox(name='pedestrians')
dataset.scan_image_directory(IMAGE_DIR)


if USE_MTURK:
  crowdsource = MTurkCrowdsourcer(dataset, AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY, HOST, OUTPUT_FOLDER, sandbox=SANDBOX,
                                  hit_params = PARAMS, online = ONLINE, thumbnail_size = (100,100), initial_assignments_per_image=WORKERS_PER_IMAGE)
else:
  crowdsource = LocalCrowdsourcer(dataset, HOST, OUTPUT_FOLDER, hit_params = PARAMS, online = ONLINE, thumbnail_size = (100,100), initial_assignments_per_image=WORKERS_PER_IMAGE, port=8080)
  
crowdsource.run()
