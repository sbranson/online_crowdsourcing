import numpy as np
import ntpath
import os
import json

class CrowdImage(object):
  def __init__(self, id, params):
    self.id = id
    self.params = params
    self.y = None      # Ground truth label, or prediction of ground truth label
    self.x = None      # Image for computer vision
    self.z = {}      # Worker annotations
    self.d = None      # difficulty parameters
    self.finished = False   # did the annotations for this image pass the minimum risk test?
    self.workers = []
    self.cv_pred = None
    self.encode_exclude = {'y':True, 'y_gt':True, 'z':True, 'cv_pred':True, 'params':True, 'encode_exclude':True}

  def crowdsource_simple(self, avoid_if_finished=False):
    return
    
  def compute_log_likelihood(self):
    return 0

  def predict_true_labels(self, avoid_if_finished=False):
    return
    
  def estimate_parameters(self):
    return

  def check_finished(self, set_finished=True):
    return
  
  def filename(self):
    return ntpath.basename(self.url) #self.original_name.replace('-', '').replace('vibe:', '')+'.jpg'
  
  def num_annotations(self):
    num = 0
    if not self.z is None:
      has_cv = (1 if (self.params.cv_worker and self.params.cv_worker.id in self.z) else 0)
      num += int(len(self.z)-has_cv)
    return num
        
  def parse(self, data):
    for k in data: 
      setattr(self, k, data[k])
  
  def encode(self):
    data = {}
    for k in self.__dict__:
      if not k in self.encode_exclude:
        data[k] = self.__dict__[k]
    return data

  def copy_parameters_from(self, image, full=True):
    if hasattr(image, 'url'): self.url = image.url
    if hasattr(image, 'fname'): self.fname = image.fname

class CrowdWorker(object):
  def __init__(self, id, params):
    self.id = id
    self.params = params
    self.images = {}  # set of images annotated by this worker
    self.encode_exclude = {'images':True, 'params':True, 'encode_exclude':True}
    self.is_computer_vision = False

  def compute_log_likelihood(self):
    return 0
  
  def estimate_parameters(self):
    return 0
  
  def parse(self, data):
    for k in data: 
      setattr(self, k, data[k])
  
  def encode(self):
    data = {}
    for k in self.__dict__:
      if not k in self.encode_exclude:
        data[k] = self.__dict__[k]
    return data
    
  def copy_parameters_from(self, worker, full=True):
    return

class CrowdLabel(object):
  def __init__(self, image, worker):
    self.image = image
    self.worker = worker
    self.encode_exclude = {'worker':True, 'image':True, 'encode_exclude':True, 'raw_data':True}
  
  def compute_log_likelihood(self):
    return 0
  
  def loss(self, y):
    return 1
  
  def copy_into(self, into):
    for attr in dir(self):
      if not callable(attr) and not attr.startswith("__") and attr != "image" and attr != "worker":
        setattr(into, attr, getattr(self, attr))
  
  def estimate_parameters(self):
    return 0

  def is_computer_vision(self):
    return self.worker and self.worker.is_computer_vision
  
  def parse(self, data):
    self.raw_data = data
    for k in data: 
      setattr(self, k, data[k])
  
  def encode(self):
    data = {}
    for k in self.__dict__:
      if not k in self.encode_exclude:
        data[k] = self.__dict__[k]
    return data

class CrowdDataset(object):
  def __init__(self, debug=0, learn_worker_params=True, learn_image_params=True, computer_vision_predictor=None, image_dir=None, min_risk=0.005, estimate_priors_automatically=False, name="", naive_computer_vision=False):
    self.debug = debug
    self.name = name
    self.min_risk = min_risk
    self.learn_worker_params = learn_worker_params
    self.learn_image_params = learn_image_params
    self.computer_vision_predictor = computer_vision_predictor
    self.image_dir = image_dir
    self.estimate_priors_automatically = estimate_priors_automatically
    self.finished = False
    self.encode_exclude = {'workers':True, 'images':True, 'cv_worker':True, '_CrowdImageClass_':True, '_CrowdWorkerClass_':True, '_CrowdLabelClass_':True, 'computer_vision_predictor':True, 'encode_exclude':True}
    self.naive_computer_vision = naive_computer_vision
    self.cv_iter = 0
    self.cv_worker = None
    self.add_computer_vision_to_workers = True
    self.images, self.workers = {}, {}

  # Set ground truth part and bounding box locations equal to the consensus median
  def crowdsource_simple(self, avoid_if_finished=False):
    for i in self.images:
      self.images[i].crowdsource_simple(avoid_if_finished=avoid_if_finished)

  # Estimate priors globally over the whole dataset
  def estimate_priors(self):
    return

  def NewCrowdLabel(self, i, w):
    return 
  
  def initialize_parameters(self, avoid_if_finished=False):
    return 
    
  def compute_log_likelihood(self):
    ll = 0
    for i in self.images:  
      ll += self.images[i].compute_log_likelihood()
    for w in self.workers:  
      ll += self.workers[w].compute_log_likelihood()
    for i in self.images:  
      for w in self.images[i].z:
        ll += self.images[i].z[w].compute_log_likelihood()
    return ll
  
  def estimate_parameters(self, max_iters=10, avoid_if_finished=False, updateProgress=False):
    self.crowdsource_simple(avoid_if_finished=avoid_if_finished)
    if not self.computer_vision_predictor is None:
      self.initialize_parameters(avoid_if_finished=avoid_if_finished)
      for i in self.images: self.images[i].predict_true_labels(avoid_if_finished=avoid_if_finished)
      self.get_computer_vision_probabilities()  
    if self.estimate_priors_automatically:
      self.estimate_priors()
    #print self.prob_fp
    self.initialize_parameters(avoid_if_finished=avoid_if_finished)
    #print 'here ' + str(self.cv_worker.__dict__)
    
    #print "worker ids"
    #for w in self.workers.values():
    #  print w.id

    if updateProgress: 
      from celery import task, current_task, Celery

    log_likelihood, old_likelihood = -np.inf, -np.inf#self.compute_log_likelihood()
    for it in range(max_iters):
      if updateProgress: current_task.update_state(state='PROGRESS', meta={"iter":(it+1),"maxIters":(max_iters+1)})
      if self.debug > 1:
        print "Estimate params for " + self.name + ", iter " + str(it+1) + " likelihood=" + str( log_likelihood)

      # Estimate part predictions in each image using worker labels and current worker parameters
      for i in self.images:
        #print "Image: %s" % (str(i),)  
        self.images[i].predict_true_labels(avoid_if_finished=avoid_if_finished)

      # Estimate difficulty parameters for each image
      if self.learn_image_params:
        for i in self.images:  
          self.images[i].estimate_parameters(avoid_if_finished=avoid_if_finished)

      # Estimate response probability parameters for each worker
      if self.learn_worker_params:
        for w in self.workers:  
          self.workers[w].estimate_parameters()

      # Estimate response probability parameters for each worker
      for i in self.images: 
        #print "OC: Working on image: ", i 
        for w in self.images[i].z:
          #print "OC: Working on worker: ", w
          self.images[i].z[w].estimate_parameters()
          #print

      # Check the new log likelihood of the dataset and finish on convergence
      log_likelihood = self.compute_log_likelihood()
      if log_likelihood <= old_likelihood:
        break
      old_likelihood = log_likelihood
  
    return log_likelihood
  
  def get_computer_vision_probabilities(self, method='at_least_one_worker'):
    ids = [i for i in self.images]
    images = [self.images[i] for i in ids]
    labels = [self.images[i].y for i in ids]
    if method=='at_least_one_worker':
      valid_train = [(len(self.images[i].z)-(1 if (self.cv_worker and self.cv_worker.id in self.images[i].z) else 0))>0 for i in ids]
    elif method=='is_finished':
      valid_train = [self.images[i].finished for i in ids]
      
    prev_cv_worker = self.cv_worker
    self.cv_worker = self._CrowdWorkerClass_('computer_vision_iter'+str(self.cv_iter), self)
    self.cv_worker.is_computer_vision = True
    self.cv_iter = self.cv_iter+1
    
    cv_preds = self.computer_vision_predictor.predict_probs(images, labels, valid_train=valid_train, cache_name=self.fname+'.computer_vision_cache', cv_worker=self.cv_worker, naive=self.naive_computer_vision)
    
    cv_preds_dict = {label.image.id : label for label in cv_preds}
    cv_preds = [cv_preds_dict[image_id] for image_id in ids]
    
    for i in range(len(ids)):
      self.images[ids[i]].cv_pred = cv_preds[i]
      self.cv_worker.images[ids[i]] = self.images[ids[i]]
    
    if self.add_computer_vision_to_workers: 
      if prev_cv_worker:
        del self.workers[prev_cv_worker.id]
      for i in range(len(ids)):  
        if prev_cv_worker and prev_cv_worker.id in self.images[ids[i]].z:
          del self.images[ids[i]].z[prev_cv_worker.id]
        self.images[ids[i]].z[self.cv_worker.id] = cv_preds[i]
        worker_ind = -1
        for j in range(len(self.images[ids[i]].workers)):
          if prev_cv_worker and self.images[ids[i]].workers[j] == prev_cv_worker.id:
            worker_ind = j
        if worker_ind >= 0:
          self.images[ids[i]].workers[worker_ind] = self.cv_worker.id
        else:
          self.images[ids[i]].workers.append(self.cv_worker.id)
      self.workers[self.cv_worker.id] = self.cv_worker
      
    
    

  def check_finished_annotations(self, set_finished=True):
    finished = {}
    for i in self.images:
      finished[self.images[i].id] = self.images[i].check_finished(set_finished=set_finished)
    return finished
    
  def num_unfinished(self, max_annos=float('Inf'), full_dataset=None):
    num = 0
    for i in self.images:
      has_cv = (1 if (self.cv_worker and self.cv_worker.id in self.images[i].z) else 0)
      fd_has_cv = (1 if (full_dataset and full_dataset.cv_worker and full_dataset.cv_worker.id in full_dataset.images[i].z) else 0)
      num_annos_available = (min(max_annos,len(full_dataset.images[i].z)-fd_has_cv) if not full_dataset is None else max_annos)
      num += int((0 if self.images[i].z is None else len(self.images[i].z)-has_cv) < num_annos_available and not self.images[i].finished)
    return num
  
  def num_annotations(self):
    num = 0
    for i in self.images:
      num += self.images[i].num_annotations()
    return num
  
  def risk(self, images=None):
    r = 0
    for i in self.images:
      r += self.images[i].risk
    return r/len(self.images)
    
  def choose_images_to_annotate_next(self, max_workers_per_image=1, sort_method="num_annos", full_dataset=None):
    if sort_method=="num_annos":
      queue = sorted(self.images.items(), key=lambda x: len(x[1].z))   # TODO: more intelligent selection mechanism?
    elif sort_method=="risk":
      queue = sorted(self.images.items(), key=lambda x: (-x[1].risk if hasattr(x[1],"risk") else len(x[1].z)))   
    elif sort_method=="normalized_risk":
      queue = sorted(self.images.items(), key=lambda x: (-x[1].risk/len(x[1].z) if len(x[1].z)>0and hasattr(x[1],"risk") else len(x[1].z)))  
    else:
      queue = full_dataset.images.items()
    
    image_ids = []
    for iq in queue:
      i = iq[0]
      if not self.images[i].finished:
        image_ids.append(i)
    return image_ids

  
  # Get the average per-part errors compared to ground truth
  def compute_error(self, gt_dataset, **kwds):
    err = 0
    num_images = 0
    for i in gt_dataset.images:
      self.images[i].loss = 0
      y = gt_dataset.images[i].y_gt if hasattr(gt_dataset.images[i], 'y_gt') else gt_dataset.images[i].y
      if not y is None:
        self.images[i].y_gt = y
        self.images[i].loss = self.images[i].y.loss(y, **kwds)
        err += self.images[i].loss
        num_images += 1
    return err / float(num_images)
  
  def load(self, fname, max_assignments=None):
    self.fname = fname
    with open(fname) as f:
      data = json.load(f)
    self.images = {}
    self.workers = {}
    if 'dataset' in data:
      self.parse(data['dataset'])
    if 'workers' in data:
      for w in data['workers']:
        self.workers[w] = self._CrowdWorkerClass_(w, self)
        self.workers[w].parse(data['workers'][w])
    if 'images' in data:
      for i in data['images']:
        self.images[i] = self._CrowdImageClass_(i, self)
        self.images[i].parse(data['images'][i])
    if 'annos' in data:
      for l in data['annos']:
        i, w, a = l['image_id'], l['worker_id'], l['anno']
        if not i in self.images:
          self.images[i] = self._CrowdImageClass_(i, self)
        if not w in self.workers:
          self.workers[w] = self._CrowdWorkerClass_(w, self)
        has_cv = (1 if (self.cv_worker and self.cv_worker.id in self.images[i].z) else 0)
        if max_assignments == None or len(self.images[i].z)-has_cv < max_assignments:
          z = self._CrowdLabelClass_(self.images[i], self.workers[w])
          z.parse(a)
          self.images[i].z[w] = z
          self.images[i].workers.append(w)
          self.workers[w].images[i] = self.images[i]
    if 'gt_labels' in data:
      for l in data['gt_labels']:
        i, a = l['image_id'], l['label']
        self.images[i].y_gt = self._CrowdLabelClass_(self.images[i], None)
        self.images[i].y_gt.parse(a)
        self.images[i].y = self.images[i].y_gt
    if 'combined_labels' in data:
      for l in data['combined_labels']:
        i, a = l['image_id'], l['label']
        self.images[i].y = self._CrowdLabelClass_(self.images[i], None)
        self.images[i].y.parse(a)
  
  def save(self, fname, save_dataset=True, save_images=True, save_workers=True, save_annos=True, save_gt_labels=True, save_combined_labels=True):
    data = {}
    if save_dataset:
      data['dataset'] = self.encode()
    if save_images:
      data['images'] = {}
      for i in self.images:
        data['images'][i] = self.images[i].encode()
    if save_workers:
      data['workers'] = {}
      for w in self.workers:
        data['workers'][w] = self.workers[w].encode()
    if save_annos:
      data['annos'] = []
      for i in self.images:
        for w in self.images[i].z:
          data['annos'].append({'image_id':i, 'worker_id':w, 'anno': self.images[i].z[w].encode()})
    if save_gt_labels:
      data['gt_labels'] = []
      for i in self.images: 
        if hasattr(self.images[i],'y_gt') and self.images[i].y_gt:
          data['gt_labels'].append({'image_id':i, 'label':self.images[i].y_gt.encode()})
    if save_combined_labels:
      data['combined_labels'] = []
      for i in self.images:
        if hasattr(self.images[i],'y') and self.images[i].y:
          data['combined_labels'].append({'image_id':i, 'label':self.images[i].y.encode()})
    with open(fname, 'w') as f:
      json.dump(data, f)
    
  def parse(self, data):
    for k in data: 
      setattr(self, k, data[k])
  
  def encode(self):
    data = {}
    for k in self.__dict__:
      if not k in self.encode_exclude:
        data[k] = self.__dict__[k]
    return data    
  
  def scan_image_directory(self, dir_name):
    print 'Scanning images from ' + dir_name + '...'
    images = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
    if not hasattr(self, 'images'):
      self.images = {}
    if not hasattr(self, 'workers'):
      self.workers = {}
    for f in images:
      i,ext = os.path.splitext(f)
      self.images[i] = self._CrowdImageClass_(i, self)
      self.images[i].fname = os.path.join(dir_name, f)
  
  def copy_parameters_from(self, dataset, full=True):
    if hasattr(dataset, 'fname'):
      self.fname = dataset.fname


def dict_copy(obj, types=(int, str, bool, float)):
  d = obj.__dict__
  data = {}
  for k in d:
    if type(d[k]) in types:
      data[k] = d[k]
  return data
