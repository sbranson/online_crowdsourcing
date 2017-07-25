from ...online_crowdsourcing import *
from part import *
import json
import math
import os
import pickle
from PIL import Image

PART_COLORS =  [ '#FF0000', '#00FF00', '#008000', '#FFBF4A', '#000080', '#FFFF00', '#626200', '#00FFFF', '#006262', '#FF00FF', '#620062', '#FFFFFF', '#000000', '#44200F' ]
PART_OUTLINE_COLORS =  [ '#000000', '#FFFFFF', '#000000', '#FFFFFF', '#000000', '#000000', '#FFFFFF', '#000000', '#FFFFFF', '#000000', '#FFFFFF', '#000000', '#FFFFFF', '#FFFFFF' ]
PART_OUTLINE_GT_COLORS = [ '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#0000FF' ]
PART_OUTLINE_UNFINISHED_COLORS = [ '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000' ]
PART_OUTLINE_FINISHED_COLORS = [ '#00FF00', '#00FF00', '#00FF00', '#00FF00', '#00FF00', '#00FF00', '#00FF00', '#00FF00', '#00FF00', '#00FF00', '#00FF00', '#00FF00', '#00FF00', '#00FF00' ]
NUM_COLS = 4

# Crowdsourcing a collection of P parts.  Many of the computations are based on CrowdDatasetPart, which handles a single part
# independently from all other parts; however, the computer vision module models correlation between parts
class CrowdDatasetParts(CrowdDataset):
  def __init__(self, part_names=None, **kwds):
    super(CrowdDatasetParts, self).__init__(**kwds)
    self._CrowdImageClass_ = CrowdImageParts
    self._CrowdWorkerClass_ = CrowdWorkerParts
    self._CrowdLabelClass_ = CrowdLabelParts
    self.opts = kwds
    
    self.part_names = part_names
    if not part_names is None:
      self.parts = [CrowdDatasetPart(p, color=PART_COLORS[p%len(PART_COLORS)], outline_color=PART_OUTLINE_GT_COLORS[p%len(PART_OUTLINE_COLORS)], name=part_names[p], **kwds) for p in range(len(self.part_names))]
    
    self.encode_exclude['parts'] = True
    self.encode_exclude['opts'] = True
    self.skill_names = ['Location Sigma', 'Prob Mistake', 'Prob Vis Correct', 'Prob Vis Correct Given Vis', 'Prob Vis Correct Given Not Vis']
    
    name = self.name if self.name and len(self.name) > 0 else "objects"
    self.hit_params = {'object_name':name};
    dollars_per_hour, sec_per_click, sec_per_hour = 8, 2, 3600
    self.reward = 0.15
    self.images_per_hit = 1
    self.reward = math.ceil(100*float(self.images_per_hit)*dollars_per_hour/sec_per_hour*sec_per_click*(len(part_names) if part_names else 10))/100.0
    self.description = self.title = "Click on parts of " + name + " in images"
    self.keywords = "click,parts,images," + name
    self.html_template_dir = 'html/parts'

  def NewCrowdLabel(self, i, w):
    return CrowdLabelParts(self.images[i], self.workers[w])
  
  def estimate_priors(self, avoid_if_finished=False):
    for p in range(len(self.parts)):
      self.parts[p].estimate_priors(avoid_if_finished=avoid_if_finished)
  
  def initialize_parameters(self, avoid_if_finished=False):
    for p in range(len(self.parts)):
      self.parts[p].initialize_parameters(avoid_if_finished=avoid_if_finished)
  
  def copy_parameters_from(self, dataset, full=True):
    super(CrowdDatasetParts, self).copy_parameters_from(dataset, full=full)
    self.part_names = dataset.part_names
    self.parts = [CrowdDatasetPart(p, color=PART_COLORS[p%len(PART_COLORS)], outline_color=PART_OUTLINE_GT_COLORS[p%len(PART_OUTLINE_COLORS)], name=self.part_names[p], **self.opts) for p in range(len(self.part_names))]
    self.reward = dataset.reward
    for p in range(len(self.parts)):
      self.parts[p].copy_parameters_from(dataset.parts[p], full=full)

  def num_unfinished(self, max_annos=float('Inf'), full_dataset=None):
    num = 0
    for p in range(len(self.parts)):
      num += self.parts[p].num_unfinished(max_annos=max_annos, full_dataset=full_dataset)
    return num

  def num_annotations(self):
    num = 0
    for p in range(len(self.parts)):
      num += self.parts[p].num_annotations()
    return num/float(len(self.parts))
  
  def risk(self, images=None):
    r = 0
    for p in range(len(self.parts)):
      r += self.parts[p].risk()
    return r
  
  def parse(self, data):
    super(CrowdDatasetParts, self).parse(data)
    self.parts = [CrowdDatasetPart(p=p, name=self.part_names[p], color=PART_COLORS[p%len(PART_COLORS)], outline_color=PART_OUTLINE_GT_COLORS[p%len(PART_OUTLINE_COLORS)]) for p in range(len(self.part_names))]
    if 'parts' in data:
      for p in range(len(self.parts)):
        self.parts[p].parse(data['parts'][p])
      
  def encode(self):
    enc = super(CrowdDatasetParts, self).encode()
    enc['parts'] = [self.parts[p].encode() for p in range(len(self.parts))]
    return enc

class CrowdImageParts(CrowdImage):
  def __init__(self, id, params):
    super(CrowdImageParts, self).__init__(id, params)
    self.bbox, self.worker_bboxes = None, {}
    if hasattr(params, 'parts') and params.parts:
      self.parts = []
      for p in range(len(params.parts)):
        self.parts.append(CrowdImagePart(id, params.parts[p], p))
        params.parts[p].images[id] = self.parts[p]
    self.encode_exclude['parts'] = True

  def crowdsource_simple(self, avoid_if_finished=False):
    if avoid_if_finished and self.finished:
      return
    
    # Take the "median" bounding box
    if len(self.worker_bboxes) > 0:
      best = float('-inf')
      best_j = -1
      for wj in self.worker_bboxes:
        sumA = 0
        if self.worker_bboxes[wj][2] == 1e-7 or self.worker_bboxes[wj][4] <= 1: sumA=-1+self.worker_bboxes[wj][4]*1e-5
        for wk in self.worker_bboxes:
          if self.worker_bboxes[wk][2] == 1e-7 or self.worker_bboxes[wk][4] <= 1:
            continue
          ux = max(self.worker_bboxes[wj][0]+self.worker_bboxes[wj][2],self.worker_bboxes[wk][0]+self.worker_bboxes[wk][2])-min(self.worker_bboxes[wj][0],self.worker_bboxes[wk][0])
          uy = max(self.worker_bboxes[wj][1]+self.worker_bboxes[wj][3],self.worker_bboxes[wk][1]+self.worker_bboxes[wk][3])-min(self.worker_bboxes[wj][1],self.worker_bboxes[wk][1])
          ix = max(0,min(self.worker_bboxes[wj][0]+self.worker_bboxes[wj][2],self.worker_bboxes[wk][0]+self.worker_bboxes[wk][2])-max(self.worker_bboxes[wj][0],self.worker_bboxes[wk][0]))
          iy = max(0,min(self.worker_bboxes[wj][1]+self.worker_bboxes[wj][3],self.worker_bboxes[wk][1]+self.worker_bboxes[wk][3])-max(self.worker_bboxes[wj][1],self.worker_bboxes[wk][1]))
          sumA += ix*iy/max(ux*uy,1e-7)
        if sumA > best:
          best = sumA
          best_j = wj
      self.bbox = self.worker_bboxes[best_j]
    
    self.y = CrowdLabelParts(self, None) 
    for p in range(len(self.parts)):
      if self.bbox: self.parts[p].set_bbox(self.bbox)
      self.parts[p].crowdsource_simple(avoid_if_finished=avoid_if_finished)
      self.y.parts[p] = self.parts[p].y

  def predict_true_labels(self, avoid_if_finished=False):
    self.y = CrowdLabelParts(self, None) 
    for p in range(len(self.parts)):
      self.parts[p].predict_true_labels(avoid_if_finished=avoid_if_finished)
      self.y.parts[p] = self.parts[p].y
  
  def compute_log_likelihood(self):
    ll = 0
    for p in range(len(self.parts)):
      ll += self.parts[p].compute_log_likelihood()
    return ll
  
  # Estimate difficulty parameters
  def estimate_parameters(self, avoid_if_finished=False):
    for p in range(len(self.parts)):
      self.parts[p].estimate_parameters(avoid_if_finished=avoid_if_finished)

  
  def check_finished(self, set_finished=True):
    finished = True
    self.risk = 0
    for p in range(len(self.parts)):
      if not self.parts[p].check_finished(set_finished=set_finished):
        finished = False
      if hasattr(self.parts[p], "risk"): 
        self.risk += self.parts[p].risk
    if set_finished: self.finished = finished
    return finished
  
  def num_annotations(self):
    num = 0
    for p in range(len(self.parts)):
      num += self.parts[p].num_annotations()
    return num/float(len(self.parts))
    
  def parse(self, data):
    super(CrowdImageParts, self).parse(data)
    self.parts = [CrowdImagePart(self.id, self.params.parts[p], p=p) for p in range(len(self.params.parts))]
    for p in range(len(self.parts)):
      if 'parts' in data:
        self.parts[p].parse(data['parts'][p])
      self.params.parts[p].images[id] = self.parts[p]
      
  def encode(self):
    enc = super(CrowdImageParts, self).encode()
    enc['parts'] = [self.parts[p].encode() for p in range(len(self.parts))]
    return enc

class CrowdWorkerParts(CrowdWorker):
  def __init__(self, id, params):
    super(CrowdWorkerParts,self).__init__(id, params)
    if hasattr(params, 'parts') and params.parts:
      self.parts = []
      for p in range(len(params.parts)):
        self.parts.append(CrowdWorkerPart(id, params.parts[p], p))
        params.parts[p].workers[id] = self.parts[p]
    self.encode_exclude['parts'] = True

  def compute_log_likelihood(self):
    ll = 0
    for p in range(len(self.parts)):
      ll += self.parts[p].compute_log_likelihood()
    return ll
  
  def estimate_parameters(self):
    for p in range(len(self.parts)):
      self.parts[p].estimate_parameters()
    self.skill = np.asarray([p.skill for p in self.parts]).mean(axis=0).tolist()
  
  def parse(self, data):
    super(CrowdWorkerParts, self).parse(data)
    self.parts = [CrowdWorkerPart(self.id, self.params.parts[p], p=p) for p in range(len(self.params.parts))]
    for p in range(len(self.parts)):
      if 'parts' in data:
        self.parts[p].parse(data['parts'][p])
      self.params.parts[p].workers[id] = self.parts[p]
      
  def encode(self):
    enc = super(CrowdWorkerParts, self).encode()
    enc['parts'] = [self.parts[p].encode() for p in range(len(self.parts))]
    return enc

class CrowdLabelParts(CrowdLabel):
  def __init__(self, image, worker):
    super(CrowdLabelParts, self).__init__(image, worker)
    #self.parts = [None for p in range(len(image.parts))]
    self.encode_exclude['parts'] = True
    self.parts = [CrowdLabelPart(image.parts[p], worker.parts[p] if worker else None, p) for p in range(len(image.parts))]
    self.gtype = 'keypoints'

  def compute_log_likelihood(self):
    ll = 0
    for p in range(len(self.parts)):
      if not self.parts[p] is None:
        ll += self.parts[p].compute_log_likelihood()
    return ll
    
  def loss(self, y):
    loss = 0
    for p in range(len(self.parts)):
      self.parts[p].image.loss = self.parts[p].loss(y.parts[p])
      loss += self.parts[p].image.loss
    loss /= float(len(self.parts))
    self.image.loss = loss
    return loss
  
  def estimate_parameters(self, avoid_if_finished=False):
    for p in range(len(self.parts)):
      if not self.parts[p] is None:
        self.parts[p].estimate_parameters(avoid_if_finished=avoid_if_finished)
        
  def parse(self, data):
    super(CrowdLabelParts, self).parse(data)
    self.parts = [CrowdLabelPart(self.image.parts[p], self.worker.parts[p] if self.worker else None, p=p) for p in range(len(data['parts']))]
    bbox = [float("inf"), float("inf"), float("-inf"), float("-inf"), 0]
    num_vis = 0
    #print str(data)
    for p in range(len(self.parts)):
      if not self.parts[p].image.finished: 
        if self.worker:# and not self.worker.id in self.worker.parts[p].params.workers:
          self.worker.parts[p].params.workers[self.worker.id] = self.worker.parts[p]
        if self.worker:# and not self.worker.id in self.worker.params.parts[p].workers:
          self.worker.params.parts[p].workers[self.worker.id] = self.worker.parts[p]
        #if not self.image.id in self.image.parts[p].params.images:
        self.image.parts[p].params.images[self.image.id] = self.image.parts[p]
        #if not self.image.id in self.image.params.parts[p].images:
        self.image.params.parts[p].images[self.image.id] = self.image.parts[p]
        if self.worker:
          self.parts[p].image.workers.append(self.worker.id)
          self.parts[p].worker.images[self.image.id] = self.image.parts[p]
        self.parts[p].parse(data['parts'][p])
        if self.worker:
          self.image.parts[p].z[self.worker.id] = self.parts[p]
        elif self == self.image.y_gt:
          self.image.parts[p].y_gt = self.parts[p]
        elif self == self.image.y:
          self.image.parts[p].y = self.parts[p]
      if data['parts'][p]['vis']:
        bbox[0] = float(min(bbox[0], data['parts'][p]['x']))
        bbox[1] = float(min(bbox[1], data['parts'][p]['y']))
        bbox[2] = float(max(bbox[2], data['parts'][p]['x']))
        bbox[3] = float(max(bbox[3], data['parts'][p]['y']))
        num_vis += 1
        
    if num_vis == 0:
      bbox = [0, 0, 1e-7, 1e-7, 0]
    else:
      bbox[0] = max(0, bbox[0])
      bbox[1] = max(0, bbox[1])
      bbox[2] = max(1e-7, bbox[2]-bbox[0])
      bbox[3] = max(1e-7, bbox[3]-bbox[1])
      bbox[4] = num_vis
    if self.worker: 
      self.image.worker_bboxes[self.worker.id] = bbox
    for p in range(len(self.parts)): 
      if not self.parts[p].image.finished:
        self.parts[p].set_bbox(bbox)

  def encode(self):
    enc = super(CrowdLabelParts, self).encode()
    enc['parts'] = [self.parts[p].encode() for p in range(len(self.parts))]
    return enc
  
      
