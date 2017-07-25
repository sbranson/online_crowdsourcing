from ...online_crowdsourcing import *
import json
import yaml
import math
import os
import numpy as np
import random
import urllib
import time

BASE_URL = "sbranson.no-ip.org/bluebirds"
NUM_COLS = 3

# Crowdsourcing for binary classification.  Incorporates a worker skill and image difficulty model
class CrowdDatasetBinaryClassification(CrowdDataset):
  def __init__(self, **kwds):
    super(CrowdDatasetBinaryClassification, self).__init__(**kwds)
    self._CrowdImageClass_ = CrowdImageBinaryClassification
    self._CrowdWorkerClass_ = CrowdWorkerBinaryClassification
    self._CrowdLabelClass_ = CrowdLabelBinaryClassification
    if (not 'estimate_priors_automatically' in kwds): self.estimate_priors_automatically = False #True 
    self.prob_present_given_present_beta, self.prob_not_present_given_not_present_beta, self.prob_correct_beta = 10,10,20  # beta prior; when estimating per worker examples, # training w/ the global prior
    self.prior_beta = 15
    self.prob_present_beta = 2.5
    self.prob_not_present_given_not_present_prior = self.prob_not_present_given_not_present = .8  # The prior probability an annotator says the class is not present when it is actually not present
    self.prob_present_given_present_prior = self.prob_present_given_present = .8  # The prior probability an annotator says the class is present when it is actually present
    self.prob_present_prior = self.prob_present = .5  # The prior probability that the class is present
    self.prob_correct_prior = self.prob_correct = .8
    self.skill_names = ['Prob Correct Given Present', 'Prob Correct Given Not Present']
    
    name = self.name if self.name and len(self.name) > 0 else "object"
    self.hit_params = {'object_name':name};
    dollars_per_hour, sec_per_click, sec_per_hour = 8, 1.2, 3600
    self.reward = 0.15
    self.images_per_hit = int(math.ceil(self.reward/dollars_per_hour*sec_per_hour/(sec_per_click*self.prob_present_prior)))
    self.description = self.title = "Click on images where " + ('an ' if name[0].lower() in ['a','e','i','o','u'] else 'a ') + name + " is present"
    self.keywords = "images,labelling,present," + name
    self.html_template_dir = 'html/binary'

  def copy_parameters_from(self, dataset, full=True):
    super(CrowdDatasetBinaryClassification, self).copy_parameters_from(dataset, full=full)
    if full:
      self.prob_not_present_given_not_present = dataset.prob_not_present_given_not_present
      self.prob_present_given_present = dataset.prob_present_given_present
      self.prob_present = dataset.prob_present
      self.prob_correct = dataset.prob_correct
      self.estimate_priors_automatically = False

  
  def estimate_priors(self, gt_dataset=None):    
    num_present, num_not_present = self.prior_beta*self.prob_present_prior, self.prior_beta*(1-self.prob_present_prior)
    num_not_present_given_not_present, num_present_given_not_present = self.prior_beta*self.prob_not_present_given_not_present_prior, self.prior_beta*(1-self.prob_not_present_given_not_present_prior)
    num_present_given_present, num_not_present_given_present = self.prior_beta*self.prob_present_given_present_prior, self.prior_beta*(1-self.prob_present_given_present_prior)
    self.initialize_parameters(avoid_if_finished=True)
    for i in self.images:
      has_cv = (1 if (self.cv_worker and self.cv_worker.id in self.images[i].z) else 0)
      if len(self.images[i].z)-has_cv < 1:
        continue
      if not gt_dataset is None:
        y = gt_dataset.images[i].y.label
      else:
        #self.images[i].predict_true_labels(avoid_if_finished=True)
        y = self.images[i].y.soft_label if hasattr(self.images[i].y,"soft_label") else self.images[i].y.label
      for w in self.images[i].z:
        if not self.images[i].z[w].is_computer_vision():
          z = self.images[i].z[w].label
          num_present += y
          num_not_present += 1-y
          num_not_present_given_not_present += (1-y)*(1-z)
          num_present_given_not_present += (1-y)*z
          num_present_given_present += y*z
          num_not_present_given_present += y*(1-z)
    self.prob_present = min(.9999,max(.0001,float(num_present)/max(0.0001,num_present+num_not_present)))
    self.prob_not_present_given_not_present = min(.9999,max(.0001,float(num_not_present_given_not_present)/max(0.0001,num_not_present_given_not_present+num_present_given_not_present)))
    self.prob_present_given_present = min(.9999,max(.0001,float(num_present_given_present)/max(0.0001,num_present_given_present+num_not_present_given_present)))
    self.prob_correct = min(.9999,max(.0001,float(num_present_given_present + num_not_present_given_not_present)/max(0.0001,num_present_given_present+num_not_present_given_present + num_not_present_given_not_present+num_present_given_not_present)))
    
  def initialize_parameters(self, avoid_if_finished=False):
    for w in self.workers: 
      self.workers[w].prob_not_present_given_not_present = self.prob_not_present_given_not_present
      self.workers[w].prob_present_given_present = self.prob_present_given_present
      self.workers[w].prob_present = self.prob_present
      self.workers[w].prob_correct = self.prob_correct


class CrowdImageBinaryClassification(CrowdImage):
  def __init__(self, id, params):
    super(CrowdImageBinaryClassification, self).__init__(id, params)
    self.original_name = id
  
  def crowdsource_simple(self, avoid_if_finished=False):
    if avoid_if_finished and self.finished:
      return
    num, numT = 0, 0
    for w in self.z: 
      if not self.z[w].is_computer_vision():
        num += self.z[w].label
        numT += 1
    label = (1.0 if (num>numT/2.0 or (num==numT/2.0 and random.random()>.5)) else 0.0)
    self.y = CrowdLabelBinaryClassification(self, None, label=label)
    self.y.soft_label = label if num!=numT/2.0 else .5
  
        
  
  def predict_true_labels(self, avoid_if_finished=False):
    if avoid_if_finished and self.finished:
      return
    
    if not self.cv_pred is None and not self.params.naive_computer_vision:
      ll_not_present,ll_present = math.log(1-self.cv_pred.prob), math.log(self.cv_pred.prob)
    else:
      ll_not_present,ll_present = math.log(1-self.params.prob_present), math.log(self.params.prob_present)
    for w in self.z:
      if not self.z[w].is_computer_vision() or self.params.naive_computer_vision:
        ll_present += math.log(self.z[w].worker.prob_present_given_present if self.z[w].label==1 else 1-self.z[w].worker.prob_present_given_present)
        ll_not_present += math.log(self.z[w].worker.prob_not_present_given_not_present if self.z[w].label==0 else 1-self.z[w].worker.prob_not_present_given_not_present)
    self.y = CrowdLabelBinaryClassification(self, None, label=(1.0 if (ll_present>ll_not_present or(ll_present==ll_not_present and random.random()>.5)) else 0.0))
    #ll_not_present,ll_present = ll_not_present-.5*math.log(1-self.params.prob_present), ll_present-.5*math.log(self.params.prob_present)
    self.ll_present, self.ll_not_present = ll_present, ll_not_present
    m = max(ll_present,ll_not_present)
    self.prob = math.exp(ll_present-m)/(math.exp(ll_not_present-m)+math.exp(ll_present-m))
    self.risk = self.prob*(1-self.y.label) + (1-self.prob)*self.y.label
    self.y.soft_label = self.prob              
  
  def compute_log_likelihood(self):
    y = self.y.soft_label if hasattr(self.y,"soft_label") else self.y.label
    return (1-y)*math.log(1-self.cv_pred.prob)+y*math.log(self.cv_pred.prob) if not self.cv_pred is None else (1-y)*math.log(1-self.params.prob_present)+y*math.log(1-self.params.prob_present)
  
  # Estimate difficulty parameters
  def estimate_parameters(self, avoid_if_finished=False):
    if (avoid_if_finished and self.finished) or len(self.z)<=1:
      return

  def check_finished(self, set_finished=True):
    if self.finished:
      return True
    
    self.risk = self.prob*(1-self.y.label) + (1-self.prob)*self.y.label
    finished = self.risk <= self.params.min_risk
    if set_finished: self.finished = finished
    return finished

class CrowdWorkerBinaryClassification(CrowdWorker):
  def __init__(self, id, params):
    super(CrowdWorkerBinaryClassification,self).__init__(id, params)
    self.skill = None
    self.prob_not_present_given_not_present = params.prob_not_present_given_not_present
    self.prob_present_given_present = params.prob_present_given_present
    self.prob_present = params.prob_present
    self.prob_correct = params.prob_correct

  def compute_log_likelihood(self):
    ll = ((self.params.prob_present_given_present*self.params.prob_present_given_present_beta-1)*math.log(self.prob_present_given_present) + 
          ((1-self.params.prob_present_given_present)*self.params.prob_present_given_present_beta-1)*math.log(1-self.prob_present_given_present))
    ll += ((self.params.prob_not_present_given_not_present*self.params.prob_not_present_given_not_present_beta-1)*math.log(self.prob_not_present_given_not_present) + 
           ((1-self.params.prob_not_present_given_not_present)*self.params.prob_not_present_given_not_present_beta-1)*math.log(1-self.prob_not_present_given_not_present))
    return ll
  
  # Estimate worker skill parameters
  def estimate_parameters(self, avoid_if_finished=False):
    # For each worker, we have binomial distributions for 1) the probability a worker thinks a
    # class is present if the class is present in the ground truth, 2) the probability a worker thinks a
    # class is present if the class is not present in the ground truth.  Each of these distributions has a 
    # Beta prior from the distribution of all workers pooled together
    num_present = self.params.prob_present_given_present_beta
    num_present_given_present = self.params.prob_present_given_present_beta*self.params.prob_present_given_present
    num_not_present = self.params.prob_not_present_given_not_present_beta
    num_not_present_given_not_present = self.params.prob_not_present_given_not_present_beta*self.params.prob_not_present_given_not_present
    num_worker_present = 0
    for i in self.images:
      y = self.images[i].y.soft_label if hasattr(self.images[i].y,"soft_label") else self.images[i].y.label
      num_present += y
      num_not_present += 1-y
      num_present_given_present += y*self.images[i].z[self.id].label
      num_not_present_given_not_present += (1-y)*(1-self.images[i].z[self.id].label)
      num_worker_present += self.images[i].z[self.id].label
    
    self.num_present, self.num_not_present, self.num_present_given_present, self.num_not_present_given_not_present, self.num_worker_present = num_present, num_not_present, num_present_given_present, num_not_present_given_not_present, num_worker_present

    #b = 5
    #self.prob_correct = (num_present_given_present + num_not_present_given_not_present) / (num_present+num_not_present)
    #num_present_given_present, num_present, num_not_present_given_not_present, num_not_present = num_present_given_present+self.prob_correct*b, num_present+b, num_not_present_given_not_present+self.prob_correct*b, num_not_present+b

    beta = min(self.params.prob_present_beta, num_present+num_not_present)
    self.prob_present = float(num_worker_present) / max(0.0001,len(self.images))
    num_present_given_present += self.prob_present*beta
    num_present += beta
    num_not_present_given_not_present += (1-self.prob_present)*beta
    num_not_present += beta

    self.prob_present_given_present = float(num_present_given_present) / num_present
    self.prob_not_present_given_not_present = float(num_not_present_given_not_present) / num_not_present
    self.skill = [self.prob_present_given_present, self.prob_not_present_given_not_present]

class CrowdLabelBinaryClassification(CrowdLabel):
  def __init__(self, image, worker, label=None):
    super(CrowdLabelBinaryClassification, self).__init__(image, worker)
    self.label = label
    self.gtype = 'binary'
    
  def compute_log_likelihood(self):
    y = self.image.y.soft_label if hasattr(self.image.y,"soft_label") else self.image.y.label
    z = self.label
    return (math.log(self.worker.prob_present_given_present)*y*z + math.log(self.worker.prob_not_present_given_not_present)*(1-y)*(1-z) +
            math.log(1-self.worker.prob_present_given_present)*y*(1-z) + math.log(1-self.worker.prob_not_present_given_not_present)*(1-y)*z)
    
  def loss(self, y):
    return abs(self.label-y.label)
    
  def parse(self, data):
    super(CrowdLabelBinaryClassification, self).parse(data)
    self.label = float(self.label)
