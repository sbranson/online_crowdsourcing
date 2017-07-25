import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.join(os.path.dirname(__file__),"../"))
from crowdsourcing.annotation_types.bbox import *

DATASET_DIR = "data/bbox/pedestrians"
OUTPUT_DIR = "output/bbox"

ASSIGNMENT_DATA = os.path.join(DATASET_DIR, "179_wh.json")
EXPERT_ASSIGNMENT_DATA = os.path.join(DATASET_DIR, "177(3)_wh.json")
MAX_ASSIGNMENTS_PER_IMAGE = 7
BATCH_SIZE = 1448
ESTIMATE_PRIORS = False

IMAGE_DIR = os.path.join(DATASET_DIR, "images")
TMP_DIR = os.path.join(OUTPUT_DIR, "tmp")
HTML_DIR = os.path.join(OUTPUT_DIR, "html")

if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
if not os.path.exists(HTML_DIR): os.mkdir(HTML_DIR)

#COMPUTER_VISION_URLS = ["http://127.0.0.1:8090"]  #["localhost:8090","localhost:8091","localhost:8092"]

expert_dataset = CrowdDatasetBBox()
expert_dataset.parse_dataset(EXPERT_ASSIGNMENT_DATA, expert=True)


full_dataset = CrowdDatasetBBox(debug=2, learn_worker_params=True, learn_image_params=True)
full_dataset.parse_dataset(ASSIGNMENT_DATA, max_assignments=MAX_ASSIGNMENTS_PER_IMAGE)

i = 0
dataset = CrowdDatasetBBox(debug=0, learn_worker_params=True, learn_image_params=True, min_risk=0.005)
dataset.parse_dataset(ASSIGNMENT_DATA, max_assignments=0)
'''dataset.computer_vision_handles = [pyjsonrpc.HttpClient(url = u) for u in COMPUTER_VISION_URLS]
dataset.image_dir = IMAGE_DIR
dataset.tmp_dir = TMP_DIR
dataset.setup_computer_vision_cross_validation()
'''

plot_data = {"ours-full":{"x":[],"y":[],"risk":[], "line-style":"r-", "point-style":"ro"}}
while dataset.num_unfinished(max_annos=MAX_ASSIGNMENTS_PER_IMAGE) > 0 and i < MAX_ASSIGNMENTS_PER_IMAGE:
  dataset.augment_annotations_if_necessary(full_dataset, BATCH_SIZE)
  dataset.estimate_parameters(avoid_if_finished=True)
  dataset.check_finished_annotations()
  i += 1
  err,num,risk = dataset.compute_error(expert_dataset), dataset.num_annotations(), dataset.risk()
  plot_data["ours-full"]["x"].append(num)
  plot_data["ours-full"]["y"].append(err)
  plot_data["ours-full"]["risk"].append(risk)
  print "At iter " + str(i) + " ave error is " + str(err) + " with " +  str(num) + " annos"
  dataset.save_gallery(os.path.join(HTML_DIR, 'bbox_online_'+str(i)+'worker.json'))

num_anno = [len(dataset.images[i].z) for i in dataset.images]

plot_data["majority-vote"] = {"x":[],"y":[], "line-style":"m-.", "point-style":"mv"}
plot_data["no-online"] = {"x":[],"y":[],"risk":[], "line-style":"c:", "point-style":"cs"}
plot_data["no-online-worker-skill"] = {"x":[],"y":[],"risk":[], "line-style":"y:", "point-style":"ys"}
for j in range(1,8):
  dataset = CrowdDatasetBBox(debug=0, learn_worker_params=True, learn_image_params=True)
  dataset.parse_dataset(ASSIGNMENT_DATA, max_assignments=j)
  dataset.crowdsource_simple()
  err,num = dataset.compute_error(expert_dataset), dataset.num_annotations()
  plot_data["majority-vote"]["x"].append(num)
  plot_data["majority-vote"]["y"].append(err)
  print "With " + str(j) + " workers and simple crowdsourcing, err=" + str(err)
  dataset.save_gallery(os.path.join(HTML_DIR, 'bbox_simple'+str(j)+'worker.json'))
  dataset.estimate_parameters()
  err,num = dataset.compute_error(expert_dataset), dataset.num_annotations()
  plot_data["no-online"]["x"].append(num)
  plot_data["no-online"]["y"].append(err)
  print "With " + str(j) + " workers, err=" + str(err)
  dataset.save_gallery(os.path.join(HTML_DIR, 'bbox_'+str(j)+'worker.json'))  
  fn, fp, sigma = [], [], []
  for w in dataset.workers:
    fn.append(dataset.workers[w].prob_fn)
    fp.append(dataset.workers[w].prob_fp)
    sigma.append(dataset.workers[w].sigma)
  
  dataset.learn_worker_params, dataset.learn_image_params = False, False
  dataset.estimate_parameters()
  err,num = dataset.compute_error(expert_dataset), dataset.num_annotations()
  plot_data["no-online-worker-skill"]["x"].append(num)
  plot_data["no-online-worker-skill"]["y"].append(err)
  print "With " + str(j) + " workers using just global priors, err=" + str(err)

i = 0
dataset = CrowdDatasetBBox(debug=0, learn_worker_params=False, learn_image_params=True, min_risk=0.005)
dataset.parse_dataset(ASSIGNMENT_DATA, max_assignments=0)
plot_data["no-worker-skill"] = {"x":[],"y":[],"risk":[], "line-style":"g--", "point-style":"g8"}
while dataset.num_unfinished(max_annos=MAX_ASSIGNMENTS_PER_IMAGE) > 0 and i < MAX_ASSIGNMENTS_PER_IMAGE:
  dataset.augment_annotations_if_necessary(full_dataset, BATCH_SIZE)
  dataset.estimate_parameters(avoid_if_finished=True)
  dataset.check_finished_annotations()
  i += 1
  err,num,risk = dataset.compute_error(expert_dataset), dataset.num_annotations(), dataset.risk()
  plot_data["no-worker-skill"]["x"].append(num)
  plot_data["no-worker-skill"]["y"].append(err)
  plot_data["no-worker-skill"]["risk"].append(risk)
  print "At iter " + str(i) + " ave error is " + str(err) + " with " +  str(num) + " annos"
  dataset.save_gallery(os.path.join(HTML_DIR, 'bbox_online_noworker_'+str(i)+'.json'))


i = 0
dataset = CrowdDatasetBBox(debug=0, learn_worker_params=True, learn_image_params=False, min_risk=0.005)
dataset.parse_dataset(ASSIGNMENT_DATA, max_assignments=0)
plot_data["no-image-difficulty"] = {"x":[],"y":[],"risk":[], "line-style":"k-", "point-style":"k*"}
while dataset.num_unfinished(max_annos=MAX_ASSIGNMENTS_PER_IMAGE) > 0 and i < MAX_ASSIGNMENTS_PER_IMAGE:
  dataset.augment_annotations_if_necessary(full_dataset, BATCH_SIZE)
  dataset.estimate_parameters(avoid_if_finished=True)
  dataset.check_finished_annotations()
  i += 1
  err,num,risk = dataset.compute_error(expert_dataset), dataset.num_annotations(), dataset.risk()
  plot_data["no-image-difficulty"]["x"].append(num)
  plot_data["no-image-difficulty"]["y"].append(err)
  plot_data["no-image-difficulty"]["risk"].append(risk)
  print "At iter " + str(i) + " ave error is " + str(err) + " with " +  str(num) + " annos"
  dataset.save_gallery(os.path.join(HTML_DIR, 'bbox_online_noimage_'+str(i)+'.json'))


i = 0
dataset = CrowdDatasetBBox(debug=0, learn_worker_params=False, learn_image_params=False, min_risk=0.005)
dataset.parse_dataset(ASSIGNMENT_DATA, max_assignments=0)
plot_data["no-worker-image"] = {"x":[],"y":[],"risk":[], "line-style":"b--", "point-style":"bh"}
while dataset.num_unfinished(max_annos=MAX_ASSIGNMENTS_PER_IMAGE) > 0 and i < MAX_ASSIGNMENTS_PER_IMAGE:
  dataset.augment_annotations_if_necessary(full_dataset, BATCH_SIZE)
  dataset.estimate_parameters(avoid_if_finished=True)
  dataset.check_finished_annotations()
  i += 1
  err,num,risk = dataset.compute_error(expert_dataset), dataset.num_annotations(), dataset.risk()
  plot_data["no-worker-image"]["x"].append(num)
  plot_data["no-worker-image"]["y"].append(err)
  plot_data["no-worker-image"]["risk"].append(risk)
  print "At iter " + str(i) + " ave error is " + str(err) + " with " +  str(num) + " annos"
  dataset.save_gallery(os.path.join(HTML_DIR, 'bbox_online_noimage_'+str(i)+'.json'))

'''
num = float(len(dataset.images))
handles, labels = [], []
plt.figure(1)
for n in plot_data:
  h = plt.plot([i/num for i in plot_data[n]["x"]], plot_data[n]["y"], plot_data[n]["line-style"], lw=3)
  if "point-style" in plot_data[n]:
    plt.plot([i/num for i in plot_data[n]["x"]], plot_data[n]["y"], plot_data[n]["point-style"])
  handles.append(h[0])
  labels.append(n)

plt.xlabel('# Annotations Per Image', fontsize=20)
plt.ylabel('Annotation Loss', fontsize=20)
plt.legend(handles, labels)
plt.show()
'''


plots = ["ours-full", "majority-vote", "no-online", "no-worker-image"]
num = float(len(dataset.images))
handles, labels = [], []
plt.figure(2)
plt.clf()
for n in plots:
  h = plt.semilogy([i/num for i in plot_data[n]["x"]], plot_data[n]["y"], plot_data[n]["line-style"], lw=3)
  if "point-style" in plot_data[n]:
    plt.semilogy([i/num for i in plot_data[n]["x"]], plot_data[n]["y"], plot_data[n]["point-style"])
  handles.append(h[0])
  labels.append(n)
plt.xlabel('# Annotations Per Image', fontsize=20)
plt.ylabel('Annotation Loss', fontsize=20)
a = plt.gca()
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.legend(handles, labels, prop={'size':20})
plt.savefig(os.path.join(OUTPUT_DIR, 'bbox_method_comparison.pdf'))
#plt.show()


plt.figure(3)
plt.clf()
n = "ours-full"
h1=plt.semilogy([i/num for i in plot_data[n]["x"]], plot_data[n]["y"], plot_data[n]["line-style"], lw=3)[0]
if "point-style" in plot_data[n]: plt.semilogy([i/num for i in plot_data[n]["x"]], plot_data[n]["y"], plot_data[n]["point-style"])
h2=plt.semilogy([i/num for i in plot_data[n]["x"]], [f+plot_data["no-online"]["y"][-1] for f in plot_data[n]["risk"]], 'g--', lw=3)[0]
plt.xlabel('# Annotations Per Image', fontsize=20)
plt.ylabel('Annotation Loss', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.legend([h1,h2], ['ours-full test error','ours-full predicted error'], prop={'size':20})
plt.savefig(os.path.join(OUTPUT_DIR, 'bbox_risk_estimation.pdf'))
#plt.show()

plt.figure(4)
plt.clf()
bins = [1.5,2.5,3.5,4.5,5.5,6.5,7.5]
plt.hist(num_anno, bins, histtype='bar', rwidth=0.8, color='g')
plt.xlabel('# Annotations Per Image', fontsize=20)
plt.ylabel('# Images', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.savefig(os.path.join(OUTPUT_DIR, 'bbox_annos_per_image.pdf'))
#plt.show()

fig = plt.figure(5)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(fp, fn, sigma, c='r', marker='o')
ax.set_xlabel('false neg rate (${p}_j^{\mathrm{fn}}$)', fontsize=16)
ax.set_ylabel('false pos rate (${p}_j^{\mathrm{fp}}$)', fontsize=16)
ax.set_zlabel('boundary accuracy ($\sigma$)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.savefig(os.path.join(OUTPUT_DIR, 'bbox_worker_skills.pdf'))
#fig.show()
