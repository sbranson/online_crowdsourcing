import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D


if '__file__' in globals(): sys.path.append(os.path.join(os.path.dirname(__file__),"../"))
from crowdsourcing.annotation_types.classification import *

DATASET_DIR = "data/classification/cub_40"
OUTPUT_DIR = "output/classification/cub_40"

ASSIGNMENT_DATA = os.path.join(DATASET_DIR, "class_labels_to_annotations.json")
MAX_ASSIGNMENTS_PER_IMAGE = 5
BATCH_SIZE = 108
ESTIMATE_PRIORS = False
USE_COMPUTER_VISION = True
NUM_RAND_PERMS = 5

IMAGE_DIR = os.path.join(DATASET_DIR, "images")
TMP_DIR = os.path.join(OUTPUT_DIR, "tmp")
HTML_DIR = os.path.join(OUTPUT_DIR, "html")

if not os.path.exists(os.path.dirname(OUTPUT_DIR)): os.mkdir(os.path.dirname(OUTPUT_DIR))
if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
if not os.path.exists(HTML_DIR): os.mkdir(HTML_DIR)
if not os.path.exists(OUTPUT_DIR+"/combined"): os.mkdir(OUTPUT_DIR+"/combined")



if USE_COMPUTER_VISION:
  if not os.path.exists(TMP_DIR): os.mkdir(TMP_DIR)
  if not os.path.exists(IMAGE_DIR): os.mkdir(IMAGE_DIR)

  CAFFE_ROOT = '/home/sbranson/code/other_peoples_code/caffe-master-git/caffe' #'/home/sbranson/caffe'
  MODEL_FILE = CAFFE_ROOT+'/models/bvlc_reference_caffenet/deploy.prototxt'
  PRETRAINED = CAFFE_ROOT+'/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
  USE_GPU=True
  caffe_model = FeatureExtractor(MODEL_FILE, PRETRAINED,np.load(CAFFE_ROOT + '/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), gpu=USE_GPU)

      

def generate_plots(plot_data, class_name):
  plots = ["full", "no-cv", "majority-vote", "no-online", "no-online-no-cv", "no-worker-image"]
  num = float(len(dataset.images))
  handles, labels = [], []
  plt.figure(1)
  plt.clf()
  for n in plots:
    h = plt.semilogy(plot_data[n]["x"], plot_data[n]["y"], plot_data[n]["line-style"], lw=3)
    handles.append(h[0])
    labels.append(n)
  plt.xlabel('# Annotations Per Image', fontsize=20)
  plt.ylabel('Annotation Loss', fontsize=20)
  a = plt.gca()
  plt.tick_params(axis='both', which='major', labelsize=16)
  plt.tick_params(axis='both', which='minor', labelsize=16)
  plt.legend(handles, labels, prop={'size':20})
  plt.savefig(os.path.join(OUTPUT_DIR, class_name, 'binary_method_comparison_online.pdf'))
  #plt.show()
  
  plots = ["no-online", "no-online-no-cv", "no-online-worker-image", "majority-vote"]
  num = float(len(dataset.images))
  handles, labels = [], []
  plt.figure(2)
  plt.clf()
  for n in plots:
    h = plt.semilogy(plot_data[n]["x"], plot_data[n]["y"], plot_data[n]["line-style"], lw=3)
    handles.append(h[0])
    labels.append(n)
  plt.xlabel('# Annotations Per Image', fontsize=20)
  plt.ylabel('Annotation Loss', fontsize=20)
  a = plt.gca()
  plt.tick_params(axis='both', which='major', labelsize=16)
  plt.tick_params(axis='both', which='minor', labelsize=16)
  plt.legend(handles, labels, prop={'size':20})
  plt.savefig(os.path.join(OUTPUT_DIR, class_name, 'binary_method_comparison.pdf'))

  if "risk" in plot_data[n] and len(plot_data[n]["risk"]):
    plt.figure(3)
    plt.clf()
    n = "no-online-no-cv"
    h1=plt.semilogy(plot_data[n]["x"], plot_data[n]["y"], plot_data[n]["line-style"], lw=3)[0]
    #h2=plt.semilogy(plot_data[n]["x"], [f+plot_data[n]["y"][-1] for f in plot_data[n]["risk"]], 'g--', lw=3)[0]
    h2=plt.semilogy(plot_data[n]["x"], plot_data[n]["risk"], 'g--', lw=3)[0]
    plt.xlabel('# Annotations Per Image', fontsize=20)
    plt.ylabel('Annotation Loss', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.legend([h1,h2], ['actual test error','predicted error'], prop={'size':20})
    plt.savefig(os.path.join(OUTPUT_DIR, class_name, 'binary_risk_estimation.pdf'))
    #plt.show()

  n = "full"
  plt.figure(4)
  plt.clf()
  bins = [.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5]
  plt.hist(plot_data[n]["num_anno"], bins, histtype='bar', rwidth=0.8, color='g')
  plt.xlabel('# Annotations Per Image', fontsize=20)
  plt.ylabel('# Images', fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=16)
  plt.tick_params(axis='both', which='minor', labelsize=16)
  plt.savefig(os.path.join(OUTPUT_DIR, class_name, 'binary_annos_per_image.pdf'))
  #plt.show()
  
  fig = plt.figure(5)
  plt.clf()
  plt.scatter(np.asarray(plot_data[n]["skills"])[:,0].tolist(), np.asarray(plot_data[n]["skills"])[:,1].tolist())
  plt.savefig(os.path.join(OUTPUT_DIR, class_name, 'binary_worker_skills.pdf'))
  #fig.show()
  
def average_plots(all_plot_data):
  combined_plot_data = {}
  for c in all_plot_data.items()[0][1]:
    combined_plot_data[c] = copy.deepcopy(plot_data[c])
    for f in ["x","y","risk"]:
      l = np.asarray([len(all_plot_data[cn][c]["x"]) for cn in all_plot_data]).max()
      counts = np.zeros((l))
      combined_plot_data[c][f] = np.zeros((l))
      for cn in all_plot_data:
        if f in all_plot_data[cn][c]:
          counts[:len(all_plot_data[cn][c][f])] += np.ones((len(all_plot_data[cn][c][f])))
          combined_plot_data[c][f][:len(all_plot_data[cn][c][f])] += all_plot_data[cn][c][f]
      combined_plot_data[c][f] = (combined_plot_data[c][f]/counts).tolist()
  return combined_plot_data


                                                   
all_plot_data, all_ave_plot_data = {}, {}
with open(ASSIGNMENT_DATA,"r") as f:
  json_data = json.load(f)
for class_name in json_data:
  if not os.path.exists(OUTPUT_DIR+"/"+class_name): os.mkdir(OUTPUT_DIR+"/"+class_name)
  if not os.path.exists(HTML_DIR+"/"+class_name): os.mkdir(HTML_DIR+"/"+class_name)
  
  plot_data_perms = {}
  for k in range(NUM_RAND_PERMS):
    data = [json_data[class_name]["worker_labels"][j] for j in np.random.permutation(len(json_data[class_name]["worker_labels"]))]
    expert_dataset = CrowdDatasetBinaryClassification()
    expert_dataset.parse_dataset_cub(data, expert=True, class_name=class_name)
    
    full_dataset = CrowdDatasetBinaryClassification(debug=2, learn_worker_params=True, learn_image_params=True)
    full_dataset.parse_dataset_cub(data, max_assignments=MAX_ASSIGNMENTS_PER_IMAGE, class_name=class_name)
    full_dataset.estimate_parameters()
    
    plot_data = {"full":{"x":[],"y":[],"risk":[], "line-style":"r-o", "use_cv":True, "worker_params":True, "image_params":True, "simple_crowdsourcing":False, "online":True}}
    plot_data["no-cv"] = {"x":[],"y":[],"risk":[], "line-style":"k-v", "use_cv":False, "worker_params":True, "image_params":True, "simple_crowdsourcing":False, "online":True}
    plot_data["no-worker-skill"] = {"x":[],"y":[],"risk":[], "line-style":"g--8", "use_cv":False, "worker_params":False, "image_params":True, "simple_crowdsourcing":False, "online":True}
    plot_data["no-image-difficulty"] = {"x":[],"y":[],"risk":[], "line-style":"k-*", "use_cv":False, "worker_params":True, "image_params":False, "simple_crowdsourcing":False, "online":True}
    plot_data["no-worker-image"] = {"x":[],"y":[],"risk":[], "line-style":"b--h", "use_cv":False, "worker_params":False, "image_params":False, "simple_crowdsourcing":False, "online":True}
    plot_data["no-online"] = {"x":[],"y":[],"risk":[], "line-style":"r:s", "use_cv":True, "worker_params":True, "image_params":True, "simple_crowdsourcing":False, "online":False}
    plot_data["no-online-no-cv"] = {"x":[],"y":[],"risk":[], "line-style":"c:s", "use_cv":False, "worker_params":True, "image_params":True, "simple_crowdsourcing":False, "online":False}
    
    plot_data["majority-vote"] = {"x":[],"y":[], "line-style":"m-.v", "use_cv":False, "worker_params":False, "image_params":False, "simple_crowdsourcing":True, "online":False}
    plot_data["no-online-worker-image"] = {"x":[],"y":[],"risk":[], "line-style":"y:s", "use_cv":False, "worker_params":False, "image_params":False, "simple_crowdsourcing":False, "online":False}

    for method in plot_data:
      if plot_data[method]["use_cv"] and not USE_COMPUTER_VISION: continue
      
      i = 0
      dataset = CrowdDatasetBinaryClassification(debug=0, learn_worker_params=plot_data[method]["worker_params"], learn_image_params=plot_data[method]["image_params"], min_risk=0.05)
      #dataset.copy_parameters_from(full_dataset)
      dataset.parse_dataset_cub(data, max_assignments=0, class_name=class_name)
      
      if plot_data[method]["use_cv"]:
        dataset.computer_vision_handles = caffe_model
        dataset.image_dir = IMAGE_DIR + "/" + class_name
        dataset.tmp_dir = TMP_DIR + "/" + class_name
      
      while dataset.num_unfinished(max_annos=MAX_ASSIGNMENTS_PER_IMAGE) > 0 and (i < MAX_ASSIGNMENTS_PER_IMAGE or plot_data[method]["online"]):
        dataset.estimate_priors_automatically = dataset.num_annotations() >= 2*len(dataset.images)
        dataset.augment_annotations_if_necessary(full_dataset, BATCH_SIZE if plot_data[method]["online"] else len(dataset.images))
        if plot_data[method]["simple_crowdsourcing"]:
          dataset.crowdsource_simple()
        else:
          dataset.estimate_parameters(avoid_if_finished=True)
          dataset.check_finished_annotations()
          risk = dataset.risk()
          plot_data[method]["risk"].append(risk)
        if not plot_data[method]["online"]: 
          for ii in dataset.images:
            dataset.images[ii].finished = False
        err,num = dataset.compute_error(expert_dataset), dataset.num_annotations(), 
        plot_data[method]["x"].append(num/float(len(dataset.images)))
        plot_data[method]["y"].append(err)
        i += 1
        print "Class " + class_name + " method " + method + " at iter " + str(i) + " ave error is " + str(err) + " with " +  str(num) + " annos"
        dataset.save_gallery(os.path.join(HTML_DIR, class_name, method+'_'+str(i)+'worker.json'))
      
      plot_data[method]['num_anno'] = [len(dataset.images[i].z) for i in dataset.images]
      if plot_data[method]["worker_params"]:
        plot_data[method]['skills'] = []
        for w in dataset.workers:
          plot_data[method]['skills'].append(dataset.workers[w].skill)
  
    plot_data_perms[k] = plot_data

  plot_data_ave = average_plots(plot_data_perms)

  generate_plots(plot_data_ave, class_name)
  all_plot_data[class_name] = plot_data_perms
  all_ave_plot_data[class_name] = plot_data_ave

with open(TMP_DIR + '/all_plot_data.pkl', 'wb') as f:
  pickle.dump(all_plot_data, f)

combined_plot_data = average_plots(all_ave_plot_data)
for c in plot_data:
  combined_plot_data[c] = copy.deepcopy(plot_data[c])
  for f in ["x","y","risk"]:
    l = np.asarray([len(all_ave_plot_data[cn][c]["x"]) for cn in all_ave_plot_data]).max()
    counts = np.zeros((l))
    combined_plot_data[c][f] = np.zeros((l))
    for cn in all_ave_plot_data:
      if f in all_ave_plot_data[cn][c]:
        counts[:len(all_ave_plot_data[cn][c][f])] += np.ones((len(all_ave_plot_data[cn][c][f])))
        combined_plot_data[c][f][:len(all_ave_plot_data[cn][c][f])] += all_ave_plot_data[cn][c][f]
    combined_plot_data[c][f] = (combined_plot_data[c][f]/counts).tolist()

generate_plots(combined_plot_data, "combined")
