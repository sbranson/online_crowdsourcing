from ...online_crowdsourcing import *
import json
import math
import os
import random
from PIL import Image

PROB_MAP_SIZE = 200
CONSERVE_MEMORY = False
ACCEPTABLE_STD_DEVS = 2.0
FINISHED_COLOR = '#0F0'
INCORRECT_COLOR = '#F00'

# Crowdsourcing a single part independently from all other parts, but using a model of worker skill and image difficulty
class CrowdDatasetPart(CrowdDataset):
  def __init__(self, p=None, name=None, color='#FF0000', outline_color='#FFFFFF', **kwds):
    super(CrowdDatasetPart, self).__init__(**kwds)
    self._CrowdImageClass_ = CrowdImagePart
    self._CrowdWorkerClass_ = CrowdWorkerPart
    self._CrowdLabelClass_ = CrowdLabelPart
    if (not 'estimate_priors_automatically' in kwds): self.estimate_priors_automatically = False  # True
    
    self.name = name
    self.color = color
    self.outline_color = outline_color
    self.p = p
    self.prior_sigma_v0, self.prob_vis_given_vis_beta, self.prob_vis_given_not_vis_beta, self.prob_mistake_beta, self.prior_sigma_image_v0  = 5,5,5,5,1
    self.correct_radius = None # How far can a prediction be from the ground truth location while still considering it to be correct
    self.prior_sigma = None  # If an annotator doesn't make a mistake, assume click location is Gaussian with ground truth mean and this std dev
    self.prob_mistake = None  # The probability an annotator makes a mistake (in this case, the click location is uniform in the bbox)
    self.prob_vis_given_not_vis = None  # The probability the annotator says visible when the ground truth part isn't visible
    self.prob_vis_given_vis = None  # The probability the annotator says visible when the ground truth part is visible
    self.prob_vis = None  # The probability the ground truth part is visible
    self.prob_vis, self.prob_vis_given_vis, self.prob_vis_given_not_vis = 0.8, 0.9, 0.1#0.97, 0.12
    self.prob_mistake, self.prior_sigma, self.correct_radius = 0.05, 0.1, 0.2#0.02, 0.06, 0.12
      
    self.skill_names = ['Location Sigma', 'Prob Mistake', 'Prob Vis Correct', 'Prob Vis Correct Given Vis', 'Prob Vis Correct Given Not Vis']
    
    name = self.name if self.name and len(self.name) > 0 else "objects"
    self.hit_params = {'object_name':name};
    dollars_per_hour, sec_per_click, sec_per_hour = 8, 2, 3600
    self.reward = 0.15
    self.images_per_hit = int(math.ceil(self.reward/dollars_per_hour*sec_per_hour/sec_per_click))
    self.description = self.title = "Click on the location of " + name + " in images"
    self.keywords = "click,parts,images," + name
  
  # Estimate priors globally over the whole dataset
  def estimate_priors(self, avoid_if_finished=False):
    num_v_v, num_nv_nv, num_v_nv, num_nv_v, num_total, var = 0, 0, 0, 0, 0, 0
    for i in self.images:
      if len(self.images[i].z) < 1:
        continue
      gt = self.images[i].y_gt if hasattr(self.images[i], 'y_gt') else self.images[i].y
      for w in self.images[i].z:
        part = self.images[i].z[w]
        if part.vis and gt.vis:
          d = (part.nx-gt.nx)**2 + (part.ny-gt.ny)**2
          var += min(1, d) + (self.prior_sigma**2)/self.images[i].num_est 
          num_v_v += 1.0
        elif gt.vis:
          num_nv_v += 1.0
        elif part.vis:
          num_v_nv += 1.0
        else:
          num_nv_nv += 1.0
        num_total += 1.0
    sigma = math.sqrt(var/num_v_v)
    print "num_v_v=" + str(num_v_v) + " num_nv_v=" + str(num_nv_v)+ " num_v_nv=" + str(num_v_nv)+ " num_nv_nv=" + str(num_nv_nv) + " num_total=" + str(num_total)
    
    var, num, num_mistakes = 0, 0, 0
    for i in self.images:
      if len(self.images[i].z) < 1:
        continue
      gt = self.images[i].y_gt if hasattr(self.images[i], 'y_gt') else self.images[i].y
      for w in self.images[i].z:
        part = self.images[i].z[w]
        if part.vis and gt.vis:
          prob = part.click_prob_good(gt, sigma)
          probR = part.click_prob_mistake(gt)
          num += prob/(prob+probR)
          num_mistakes += probR/(prob+probR)
          var += ((part.nx-gt.nx)**2 + (part.ny-gt.ny)**2)*prob/(prob+probR) + (self.prior_sigma**2)/self.images[i].num_est
    self.prob_vis = max((num_v_v+num_nv_v)/num_total, 1e-7)
    self.prob_vis_given_vis = max((num_v_v+self.prob_vis*self.prob_vis_given_vis_beta)/(num_v_v+num_nv_v+self.prob_vis_given_vis_beta), 1e-7)
    self.prob_vis_given_not_vis = max((num_v_nv+(1-self.prob_vis)*self.prob_vis_given_not_vis_beta)/(num_v_nv+num_nv_nv+self.prob_vis_given_not_vis_beta), 1e-7)
    self.prob_mistake = num_mistakes/num_v_v
    self.prior_sigma = max(math.sqrt(var/num), 1e-7)
    self.correct_radius = self.prior_sigma*ACCEPTABLE_STD_DEVS

    if self.debug > 0:
      print "estimate_priors for " + self.name + ":"
      print "  priorSigmas="+str(self.prior_sigma)
      print "  probMistake="+str(self.prob_mistake)
      print "  correctRadius="+str(self.correct_radius)
      print "  probVis="+str(self.prob_vis)
      print "  probVisGivenVis="+str(self.prob_vis_given_vis)
      print "  probVisGivenNotVis="+str(self.prob_vis_given_not_vis)
    
    self.initialize_parameters(avoid_if_finished=avoid_if_finished)

  def initialize_parameters(self, avoid_if_finished=False):
    # Initialize worker responses probabilities to the global priors
    for w in self.workers:
      self.workers[w].prob_vis_given_vis, self.workers[w].prob_vis_given_not_vis = self.prob_vis_given_vis, self.prob_vis_given_not_vis
      self.workers[w].prob_mistake, self.workers[w].sigma = self.prob_mistake, self.prior_sigma
    
    # Initialize image difficulties to the global priors
    for i in self.images:
      if (not avoid_if_finished or not self.images[i].finished):
        self.images[i].sigma = self.prior_sigma
      
    # Initialize image specific parameters
    for i in self.images:
      if not self.images[i].z is None and (not avoid_if_finished or not self.images[i].finished):
        for w in self.images[i].z:
          self.images[i].z[w].sigma = self.prior_sigma
  
  def copy_parameters_from(self, dataset, full=True):
    super(CrowdDatasetPart, self).copy_parameters_from(dataset, full=full)
    if full:
      self.prob_vis, self.prob_vis_given_vis, self.prob_vis_given_not_vis = dataset.prob_vis, dataset.prob_vis_given_vis, dataset.prob_vis_given_not_vis
      self.prob_mistake, self.prior_sigma = dataset.prob_mistake, dataset.prior_sigma
    self.correct_radius = dataset.correct_radius



class CrowdImagePart(CrowdImage):
  def __init__(self, id, params, p=None):
    super(CrowdImagePart, self).__init__(id, params)
    self.p = p
    self.bbox = None
    self.use_EM = True
    self.worker_probs, self.cv_probs = None, None
    if params: self.sigma = params.prior_sigma
    
  def set_bbox(self, bbox):
    if not bbox: return
    self.bbox = bbox
    if not self.z is None:
      for w in self.z:
        self.z[w].set_bbox(bbox)
    if not self.y is None:
      self.y.set_bbox(bbox)

  def crowdsource_simple(self, avoid_if_finished=False):
    if avoid_if_finished and self.finished:
      return
      
    self.set_bbox(self.bbox)

    # Make a part visible iff at least half the workers thought it was visible
    num_vis, num = 0, 0
    if not self.z is None: 
      for w in self.z:
        num_vis += float(self.z[w].vis)
        num += 1
    vis = True if (num_vis > num/2.0 or (num_vis == num/2.0 and random.random()>.5)) else False
    
    x = y = nx = ny = 0
    if num_vis > 0:
      # Find the "median" part
      self.num_est = 0
      best = float("inf")
      best_j = -1
      for wj in self.z:
        if self.z[wj].vis:
          sumD = 0
          for wk in self.z:
            if self.z[wk].vis:
              dx = self.z[wj].nx - self.z[wk].nx
              dy = self.z[wj].ny - self.z[wk].ny
              sumD += min(math.sqrt(dx*dx+dy*dy),.5)
              self.num_est += 1
          if sumD < best:
            best = sumD
            best_j = wj
            x = self.z[wj].x
            y = self.z[wj].y
            nx = self.z[wj].nx
            ny = self.z[wj].ny
    self.y = CrowdLabelPart(self, None, self.p, x=x, y=y, vis=vis, mistake=False, nx=nx, ny=ny, bbox=self.bbox)

  def predict_true_labels(self, avoid_if_finished=False):
    if avoid_if_finished and self.finished:
      return

    self.set_bbox(self.bbox)
    
    if self.y is None:
      self.y = CrowdLabelPart(self, None, self.p, x=0, y=0, vis=False, mistake=False, bbox=self.bbox)
    
    # Solve for the predicted part visibility that maximize the likelihood of the observed labels
    ll_vis = math.log(self.params.prob_vis)
    ll_not_vis = math.log(1-self.params.prob_vis)
    if not self.cv_probs is None:
      prob_vis_cv = 1-self.cv_prob_not_vis
      ll_vis += math.log(prob_vis_cv)
      ll_not_vis += math.log(1-prob_vis_cv)
    self.y.vis = True
    self.predict_mistakes()
    if not self.z is None:
      for w in self.z:
        ll_vis += self.z[w].compute_log_likelihood(simple=True) 
        self.y.vis = False
        ll_not_vis += self.z[w].compute_log_likelihood(simple=True)
        old = self.y.vis
        self.y.vis = True if ll_vis >= ll_not_vis else False
        if self.y.vis != old: self.label_changed = True
      self.y.prob_vis = math.exp(ll_vis) / (math.exp(ll_vis) + math.exp(ll_not_vis))
    
    # Solve for the predicted part location that maximizes the likelihood of the observed labels
    if self.y.vis:
      x = y = s = 0
      self.num_est = 0
      old_x,old_y = self.y.nx, self.y.ny
      if self.cv_probs is None:
        # Closed form solution without computer vision
        #print str(self.id) + " " + str(self.p)
        if not self.z is None:
          for w in self.z:
            if self.z[w].vis:
              self.num_est += 1
              g = 1-float(self.z[w].mistake)
              x += g*self.z[w].nx/(self.z[w].sigma**2)
              y += g*self.z[w].ny/(self.z[w].sigma**2)
              s += g/(self.z[w].sigma**2)
        if s > 0:
          self.y.set_normalized_loc(x/s, y/s, bbox=self.bbox)
      else:
        # probability map for combining with computer vision
        self.compute_worker_prob_map()
        self.num_est = 1
        if not self.z is None:
          for w in self.z: 
            if self.z[w].vis:
              self.num_est += 1
        xx = (np.array(range(self.worker_probs.shape[1])))/float(self.worker_probs.shape[1]-1)
        yy = (np.array(range(self.worker_probs.shape[0])))/float(self.worker_probs.shape[0]-1)
        self.map_inds = np.unravel_index((self.cv_probs_big*self.worker_probs).argmax(), self.worker_probs.shape)
        self.cv_vis_prob = self.cv_probs_big[self.map_inds[0],self.map_inds[1]]
        self.y.set_normalized_loc(xx[self.map_inds[1]], yy[self.map_inds[0]], bbox=self.bbox)
        if CONSERVE_MEMORY:
          del self.worker_probs
          del self.cv_probs_big
        if self.y.nx != old_x or self.y.ny != old_y: 
          self.label_changed = True
    
    self.predict_mistakes()
  
  def compute_log_likelihood(self):
    #print str(self.__dict__)
    # Scaled-inv-chi-squared prior for sigma parameters
    return -self.params.prior_sigma_image_v0*(self.params.prior_sigma**2)/(2*(self.sigma**2)) - (1+self.params.prior_sigma_image_v0/2)*math.log(self.sigma**2)
  
  # Estimate difficulty parameters
  def estimate_parameters(self, avoid_if_finished=False):
    if avoid_if_finished and self.finished:
      return
    
    self.set_bbox(self.bbox)
    
    # Assume each worker has their own Normal distribution defining where they click
    # with mean equal to the ground truth location.  Put a scaled inverse chi-squared prior
    # with the distribution of all workers pooled together
    numerator = self.params.prior_sigma_image_v0*(self.params.prior_sigma**2)
    denominator = 2 + self.params.prior_sigma_image_v0
    if not self.z is None:
      for w in self.z:
        if self.y.vis and self.z[w].vis:
          g = (1 - float(self.z[w].mistake)) * self.z[w].sigma_w_i
          d2 = ((self.z[w].nx-self.y.nx)**2 + (self.z[w].ny-self.y.ny)**2)
          d2 += self.params.prior_sigma**2 / (self.num_est)  # variance in estimation of pred_part_locs[p]
          numerator += g*d2
          denominator += g
    self.sigma = math.sqrt(numerator/denominator)
      
  def check_finished(self, set_finished=True):
    if self.finished:
      return True
    if False:#self.z is None or len(self.z)==0:
      if set_finished: self.finished = False
    else:
      '''
      # First compute the probability that the visibility prediction is correct
      log_prob_vis = math.log(self.params.prob_vis)# + math.log(self.prob_loc[p])
      log_prob_not_vis = math.log(1-self.params.prob_vis)
      old_vis = self.y.vis
      if not self.cv_probs is None:
        if CONSERVE_MEMORY: self.compute_worker_prob_map()
        log_prob_vis += math.log(1-self.cv_prob_not_vis)
        log_prob_not_vis += math.log(self.cv_prob_not_vis)
      if not self.z is None:
        for w in self.z:
          self.y.vis = True
          log_prob_vis += self.z[w].compute_log_likelihood(simple=True)
          self.y.vis = False
          log_prob_not_vis += self.z[w].compute_log_likelihood(simple=True)
      self.y.vis = old_vis
      self.prob_vis = math.exp(log_prob_vis) / (math.exp(log_prob_vis) + math.exp(log_prob_not_vis))
      '''
      if not self.y.vis:
        self.risk = self.y.prob_vis
      else:
        # If the part is visible, compute the probability that the predicted part location is within
        # correctRadius of the latent ground truth location
        if self.cv_probs is None:
          # Analytical solution without computer vision
          s = 0
          if not self.z is None:
            for w in self.z:
              g = 1-float(self.z[w].mistake)
              if self.z[w].vis:
                s += g / (2*(self.z[w].sigma**2))
          self.prob_loc_correct = math.erf(self.params.correct_radius*math.sqrt(s))
        else:
          # With computer vision, sum the probability map within correctRadius from the predicted location
          xx = (np.array(range(self.worker_probs.shape[1])))/float(self.worker_probs.shape[1]-1)
          yy = (np.array(range(self.worker_probs.shape[0])))/float(self.worker_probs.shape[0]-1)
          xv, yv = np.meshgrid(xx, yy)
          dists = (xv-self.y.nx)**2+(yv-self.y.ny)**2
          pc = self.worker_probs*self.cv_probs_big
          self.prob_loc_correct = pc[dists <= self.params.correct_radius**2].sum() / pc.sum()
        
        self.risk = 1-self.prob_loc_correct*self.y.prob_vis
      finished = True if self.risk < self.params.min_risk else False
      if set_finished: 
        self.finished = finished
        if self.finished:
          if not hasattr(self.y, 'original_outline_color') and hasattr(self.y, 'outline_color'): 
            self.y.original_outline_color = self.y.outline_color
          self.y.outline_color = FINISHED_COLOR
      if CONSERVE_MEMORY and not self.cv_probs is None:
        del self.worker_probs
        del self.cv_probs_big
    
    return finished
  
  # Decide whether or not each worker is making a mistake in each part click location,
  # resulting in a Uniform distribution for click location instead of Normal distribution
  def predict_mistakes(self):
    self.set_bbox(self.bbox)
    
    if self.y.vis and not self.z is None:
      for w in self.z:
        if self.z[w].vis:
          self.z[w].mistake = False
          ll_vis_good = self.z[w].compute_log_likelihood()
          self.z[w].mistake = True
          ll_vis_mistake = self.z[w].compute_log_likelihood()
          ma = max(ll_vis_good,ll_vis_mistake)
          self.z[w].mistake = math.exp(ll_vis_mistake-ma)/(math.exp(ll_vis_good-ma)+math.exp(ll_vis_mistake-ma)) if self.use_EM else ll_vis_good<ll_vis_mistake 
  
  def compute_worker_prob_map(self):
    # probability map for combining with computer vision
    a = np.array(Image.fromarray(self.cv_probs).resize((PROB_MAP_SIZE,PROB_MAP_SIZE), Image.ANTIALIAS))
    self.cv_probs_big = a*((1-self.cv_prob_not_vis)/a.sum())                                                   
    self.worker_probs = np.zeros(self.cv_probs_big.shape)
    xx = (np.array(range(self.worker_probs.shape[1])))/float(self.worker_probs.shape[1]-1)
    yy = (np.array(range(self.worker_probs.shape[0])))/float(self.worker_probs.shape[0]-1)
    if not self.z is None:
      for w in self.z:
        if self.z[w].vis:
          g = (1 - float(self.z[w].mistake))
          sigma = self.z[w].sigma
          px = np.exp( - (xx - self.z[w].nx)**2 / (2 * sigma**2)) 
          py = np.exp( - (yy - self.z[w].ny)**2 / (2 * sigma**2))
          pmis = 1.0/np.prod(self.worker_probs.shape)
          pnmis = np.outer(py,px)/(sigma * np.sqrt(2 * np.pi))
          g = pnmis / (pnmis+pmis)
          self.worker_probs += np.log((1-g)*pmis + g*pnmis)
    self.worker_probs -= self.worker_probs.max()
    self.worker_probs = np.exp(self.worker_probs)
    self.worker_probs /= self.worker_probs.sum()
        
class CrowdWorkerPart(CrowdWorker):
  def __init__(self, id, params, p=None):
    super(CrowdWorkerPart,self).__init__(id, params)
    self.p = p
    if params:
      self.prob_vis_given_vis, self.prob_vis_given_not_vis = params.prob_vis_given_vis, params.prob_vis_given_not_vis
      self.prob_mistake, self.sigma = params.prob_mistake, params.prior_sigma

  def compute_log_likelihood(self):
    # log likelihoods for Beta priors on per worker Binomial parameters
    #print str(self.__dict__)
    ll = ((self.params.prob_vis_given_vis*self.params.prob_vis_given_vis_beta-1)*math.log(self.prob_vis_given_vis) + 
          ((1-self.params.prob_vis_given_vis)*self.params.prob_vis_given_vis_beta-1)*math.log(1-self.prob_vis_given_vis))
    ll += ((self.params.prob_vis_given_not_vis*self.params.prob_vis_given_not_vis_beta-1)*math.log(self.prob_vis_given_not_vis) + 
           ((1-self.params.prob_vis_given_not_vis)*self.params.prob_vis_given_not_vis_beta-1)*math.log(1-self.prob_vis_given_not_vis))
    ll += ((self.params.prob_mistake*self.params.prob_mistake_beta-1)*math.log(self.prob_mistake) + 
           ((1-self.params.prob_mistake)*self.params.prob_mistake_beta-1)*math.log(1-self.prob_mistake))
      
    # log likelihoods for scaled inverse chi-squared priors on per worker Gaussian parameters
    ll += -self.params.prior_sigma_v0*(self.params.prior_sigma**2)/(2*(self.sigma**2)) - (1+self.params.prior_sigma_v0/2)*math.log(self.sigma**2)
    return ll
  
  def estimate_parameters(self):
    # For each worker, we have binomial distributions for 1) the probability a worker thinks a
    # part is visible if the ground truth part is visible, 2) the probability a worker thinks a
    # part is visible if the ground truth part is not visible, and 3) the probability that
    # a worker makes a mistake clicking (even though visibility was correctly predicted).
    # Each of these distributions has a Beta prior from the distribution of all workers pooled
    # together
    numGivenVis = self.params.prob_vis_given_vis_beta
    numVisGivenVis = self.params.prob_vis_given_vis_beta*self.params.prob_vis_given_vis
    numGivenNotVis = self.params.prob_vis_given_not_vis_beta
    numVisGivenNotVis = self.params.prob_vis_given_not_vis_beta*self.params.prob_vis_given_not_vis
    numCandidatesForMistakes = self.params.prob_mistake_beta
    numMistakes = self.params.prob_mistake_beta*self.params.prob_mistake
    for i in self.images:
      if self.images[i].y.vis:
        numGivenVis += 1.0
        if self.images[i].z[self.id].vis:
          numVisGivenVis += 1.0
          numCandidatesForMistakes += 1.0
          numMistakes += float(self.images[i].z[self.id].mistake)
      else:
        numGivenNotVis += 1.0
        if self.images[i].z[self.id].vis:
          numVisGivenNotVis += 1.0
    self.numVisGivenVis, self.numGivenVis, self.numVisGivenNotVis, self.numGivenNotVis, self.numMistakes, self.numCandidatesForMistakes= numVisGivenVis, numGivenVis, numVisGivenNotVis, numGivenNotVis, numMistakes, numCandidatesForMistakes
    self.prob_vis_given_vis = numVisGivenVis / numGivenVis
    self.prob_vis_given_not_vis = numVisGivenNotVis / numGivenNotVis
    self.prob_mistake = numMistakes / numCandidatesForMistakes
    self.prob_vis_correct = (numVisGivenVis+(numGivenNotVis-numVisGivenNotVis)) / max(0.0001, numGivenVis+numGivenNotVis)
    
    # Assume each worker has their own Normal distribution defining where they click
    # with mean equal to the ground truth location.  Put a scaled inverse chi-squared prior
    # with the distribution of all workers pooled together
    numerator = self.params.prior_sigma_v0*(self.params.prior_sigma**2)
    denominator = 2 + self.params.prior_sigma_v0
    for i in self.images:
      if self.images[i].y.vis and self.images[i].z[self.id].vis:
        g = (1 - float(self.images[i].z[self.id].mistake))*(1-self.images[i].z[self.id].sigma_w_i)
        numerator += g*((self.images[i].z[self.id].nx-self.images[i].y.nx)**2 + (self.images[i].z[self.id].ny-self.images[i].y.ny)**2)
        denominator += g
    self.sigma = math.sqrt(numerator/denominator)
    self.skill = [self.sigma, self.prob_mistake, self.prob_vis_correct, self.prob_vis_given_vis, 1-self.prob_vis_given_not_vis]

  def str(self):
    return (str(self.id) + ": sigma=" + str(self.sigma) + " prob_mistake=" + str(self.prob_mistake) + " prob_vis_given_vis=" + str(self.prob_vis_given_vis) + 
            " prob_vis_given_not_vis=" + str(self.prob_vis_given_not_vis) + " num=" + str(len(self.images)))

class CrowdLabelPart(CrowdLabel):
  def __init__(self, image, worker, p=None, x=None, y=None, vis=None, mistake=False, nx=None, ny=None, bbox=None):
    super(CrowdLabelPart, self).__init__(image, worker)
    self.p = p
    self.x = x
    self.y = y
    self.vis = vis
    self.mistake = mistake
    self.nx = nx
    self.ny = ny
    self.bbox = bbox
    self.correct = True
    self.sigma_w_i = .5
    self.image = image
    self.worker = worker
    self.gtype = 'keypoint'
    if worker: self.sigma = worker.sigma
    if image and image.params:
      self.color, self.outline_color, self.name = image.params.color, image.params.outline_color, image.params.name

  def set_bbox(self, bbox):
    self.bbox = bbox
    self.nx = (self.x-bbox[0])/bbox[2]
    self.ny = (self.y-bbox[1])/bbox[3]
  
  def set_loc(self, x, y):
    self.x = x
    self.y = y
    if not self.bbox is None:
      self.set_bbox(self.bbox)
  
  def set_normalized_loc(self, nx, ny, bbox=None):
    self.nx = nx
    self.ny = ny
    if bbox: self.bbox = bbox
    if not self.bbox is None:
      self.x = self.bbox[0] + self.bbox[2]*nx
      self.y = self.bbox[1] + self.bbox[3]*ny

  def click_prob_good(self, gt, dev):
    r = max(1e-8,math.sqrt((self.nx-gt.nx)**2 + (self.ny-gt.ny)**2))
    dr = math.sqrt(1.0/(self.bbox[2]**2)+1.0/(self.bbox[3]**2))
    return math.exp(-(r**2)/(2*(dev**2)))/(4*math.pi*dev)*(math.erf((r+dr)/(math.sqrt(2.0)*dev))-math.erf(r/(math.sqrt(2.0)*dev)))
  
  def click_log_prob_good(self, gt, dev):
    return math.log(max(1e-8, self.click_prob_good(gt,dev)))
  
  def click_prob_mistake(self, gt):
    return 1.0/(self.bbox[2]*self.bbox[3])
  
  def compute_log_likelihood(self, simple=False):
    gt = self.image.y
    #print str(self.__dict__)
    if gt.vis and self.vis:
      m = float(self.mistake)
      return (math.log(self.worker.prob_vis_given_vis) + (math.log(max(1e-8, (1-m)*(1-self.worker.prob_mistake)*self.click_prob_good(gt, self.sigma) + 
                                                                       m*self.worker.prob_mistake/(self.bbox[2]*self.bbox[3]))) if not simple else 0))
    elif gt.vis and (not self.vis):
      return math.log(1-self.worker.prob_vis_given_vis)
    elif (not gt.vis) and (not self.vis):
      return math.log(1-self.worker.prob_vis_given_not_vis)
    elif (not gt.vis) and self.vis:
      if self.worker.prob_vis_given_not_vis <= 0: print "bad self.worker.prob_vis_given_not_vis " + str(self.worker.__dict__)
      if self.bbox[2]*self.bbox[3] <= 0: print "bad self.bbox[2]*self.bbox[3] " + str(self.__dict__)
      return math.log(self.worker.prob_vis_given_not_vis) - (math.log(max(1e-7,self.bbox[2]*self.bbox[3])) if not simple else 0)
  
  def loss(self, y):
    self.loss2 = 1.0 if y.vis != self.vis else 0.0
    if y.vis and self.vis: 
      tmp = CrowdLabelPart(None, None, self.p, x=self.x, y=self.y, vis=self.vis)
      tmp.set_bbox(y.bbox)
      self.loss2 = 1.0 if math.sqrt((tmp.nx-y.nx)**2 + (tmp.ny-y.ny)**2) > self.image.params.correct_radius else 0
    if self.loss2 > 0:
      if not hasattr(self, 'original_outline_color'): self.original_outline_color = self.outline_color
      self.outline_color = INCORRECT_COLOR
    elif hasattr(self, 'original_outline_color'): 
      self.outline_color = FINISHED_COLOR if (self.image and self.image.finished) else self.original_outline_color
    return self.loss2
  
  def estimate_parameters(self, avoid_if_finished=False):
    # Assume each image-worker pair has its own Normal distribution defining where the worker clicks
    # with mean equal to the ground truth location.  This is a combination of image difficulty and
    # worker competence.  
    if self.image.y.vis and self.vis:
      d2 = ((self.nx-self.image.y.nx)**2 + (self.ny-self.image.y.ny)**2)
      d2 += self.image.params.prior_sigma**2 / (self.image.num_est)  # variance in estimation of pred_part_locs[p]
      pi = math.exp(-.5*d2/self.image.sigma)/self.image.sigma if self.image.params.learn_image_params else 0
      pw = math.exp(-.5*d2/self.worker.sigma)/self.worker.sigma if self.worker.params.learn_worker_params else 0
      self.sigma_w_i = pi/(pw+pi) if pw+pi>0 else 0  # Probability that variance came from image difficulty
      self.sigma = self.image.sigma*self.sigma_w_i + self.worker.sigma*(1-self.sigma_w_i)
  
  def str(self):
    return str(dict_copy(self)) 
