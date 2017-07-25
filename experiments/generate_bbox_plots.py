import sys
import os

from crowdsourcing.interfaces.simulator import *
from crowdsourcing.annotation_types.bbox import *


RAND_PERMS = 1

# ALL_METHODS = [
#   PROB_WORKER_IMAGE_CV_ONLINE, 
#   # PROB_WORKER_IMAGE_CV_NAIVE_ONLINE, 
#   # PROB_WORKER_IMAGE_ONLINE, 
#   # PROB_ONLINE, 
#   # PROB_WORKER_IMAGE_CV, 
#   # PROB_WORKER_IMAGE, 
#   PROB, 
#   SIMPLE_CROWDSOURCING
# ]

# ALL_PLOTS = [{ 
#   'title':'Method Comparison', 
#   'name':'method_comparison_semilog', 
#   'type':'semilog', 
#   'legend':True, 
#   'xlabel':'# Labels/Image', 
#   'ylabel':'Error', 
#   'methods':[
#     {
#       'name':PROB_WORKER_IMAGE_CV_ONLINE['name'],
#       'x':'num', 
#       'y':'err'
#     }, 
#     # {
#     #   'name':PROB_WORKER_IMAGE_CV_NAIVE_ONLINE['name'],
#     #   'x':'num', 
#     #   'y':'err'
#     # }, 
#     # {
#     #   'name':PROB_WORKER_IMAGE_ONLINE['name'],
#     #   'x':'num', 
#     #   'y':'err'
#     # }, 
#     # {
#     #   'name':PROB_ONLINE['name'],
#     #   'x':'num', 
#     #   'y':'err'
#     # }, 
#     # {
#     #   'name':PROB_WORKER_IMAGE_CV['name'],
#     #   'x':'num', 
#     #   'y':'err'
#     # }, 
#     # {
#     #   'name':PROB_WORKER_IMAGE['name'],
#     #   'x':'num', 
#     #   'y':'err'
#     # }, 
#     {
#       'name':PROB['name'],
#       'x':'num', 
#       'y':'err'
#     }, 
#     {
#       'name':SIMPLE_CROWDSOURCING['name'],
#       'x':'num', 
#       'y':'err'
#     }
#   ],
#   'line-width':3, 
#   'bar-color':'g', 
#   'bar-width':0.8, 
#   'axis-font-size':20, 
#   'title-font-size':30, 
#   'tick-font-size':16, 
#   'legend-font-size':20
# }]

dt = ComputerVisionDetector(num_splits=2, simulate=True)
full_dataset = CrowdDatasetBBox(computer_vision_predictor=dt, debug=2)
full_dataset.load('data/bbox/pedestrians/pedestrians.json')
sc = SimulatedCrowdsourcer(full_dataset, expert_dataset=full_dataset, num_rand_perms=1, output_dir='output/'+'bbox/pedestrians', save_prefix=ALL_METHODS[0]['name'], **ALL_METHODS[0])
sc.run()
#RunSimulatedExperiments(full_dataset, ALL_METHODS, 'bbox/pedestrians', ALL_PLOTS, title='Caltech Pedestrians', num_rand_perms=RAND_PERMS, force_compute=True)


