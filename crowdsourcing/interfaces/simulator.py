import numpy as np
import os
import sys
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.join(os.path.dirname(__file__),"../"))
from crowdsourcing.annotation_types.classification import *

def combine_dicts(ds):
  v = {}
  for d in ds:
    for k in d:
      v[k] = d[k]
  return v 

# Defining a useful suite of lesion study experiments and standardized plot styles across different annotation types
#-------------------------------------------------------
DEFAULT_PLOT_PARAMS = {'line-width':3, 'bar-color':'g', 'bar-width':0.8, 'axis-font-size':20, 'title-font-size':30, 'tick-font-size':16, 'legend-font-size':14}

PROB_WORKER_IMAGE_CV_ONLINE = {'name':'prob-worker-cv-online', 'line_style':'-', 'color':'r', 'use_computer_vision':True, 'online':True, 'simple_crowdsourcing':False, 'learn_worker_params':True, 'learn_image_params':True, 'naive_computer_vision':False, 'batch_size':1000, 'sort_method':'num_annos'}
PROB_WORKER_IMAGE_CV_NAIVE_ONLINE = {'name':'prob-worker-cv-naive-online', 'line_style':'-', 'color':'c', 'use_computer_vision':True, 'online':True, 'simple_crowdsourcing':False, 'learn_worker_params':True, 'learn_image_params':True, 'naive_computer_vision':True, 'batch_size':1000, 'sort_method':'num_annos'}
PROB_WORKER_IMAGE_ONLINE = {'name':'prob-worker-online', 'line_style':'-', 'color':'g', 'use_computer_vision':False, 'online':True, 'simple_crowdsourcing':False, 'learn_worker_params':True, 'learn_image_params':True, 'batch_size':1000, 'sort_method':'num_annos'}
PROB_ONLINE = {'name':'prob-online', 'line_style':'-', 'color':'b', 'use_computer_vision':False, 'online':True, 'simple_crowdsourcing':False, 'learn_worker_params':False, 'learn_image_params':False, 'batch_size':1000, 'sort_method':'num_annos'}

PROB_WORKER_IMAGE_CV_ONLINE_0005 = combine_dicts([PROB_WORKER_IMAGE_CV_ONLINE, {'name':'prob-worker-cv-online-.005', 'min_risk':0.005, 'color':'#FF0000'}])
PROB_WORKER_IMAGE_CV_ONLINE_001 = combine_dicts([PROB_WORKER_IMAGE_CV_ONLINE, {'name':'prob-worker-cv-online-.01', 'min_risk':0.01, 'color':'#BB0000'}])
PROB_WORKER_IMAGE_CV_ONLINE_002 = combine_dicts([PROB_WORKER_IMAGE_CV_ONLINE, {'name':'prob-worker-cv-online-.02', 'min_risk':0.02, 'color':'#770000'}])



PROB_WORKER_IMAGE_CV = {'name':'prob-worker-cv', 'line_style':'-.s', 'color':'r', 'use_computer_vision':True, 'online':False, 'simple_crowdsourcing':False, 'learn_worker_params':True, 'learn_image_params':True, 'naive_computer_vision':False, 'batch_size':1000, 'sort_method':'num_annos'}
PROB_WORKER_IMAGE = {'name':'prob-worker', 'line_style':'-.o', 'color':'g', 'use_computer_vision':False, 'online':False, 'simple_crowdsourcing':False, 'learn_worker_params':True, 'learn_image_params':True, 'batch_size':1000, 'sort_method':'num_annos'}
PROB_WORKER = {'name':'prob-worker-noim', 'line_style':'-.o', 'color':'k', 'use_computer_vision':False, 'online':False, 'simple_crowdsourcing':False, 'learn_worker_params':True, 'learn_image_params':False, 'batch_size':1000, 'sort_method':'num_annos'}
PROB = {'name':'prob', 'line_style':'-.*', 'color':'b', 'use_computer_vision':False, 'online':False, 'simple_crowdsourcing':False, 'learn_worker_params':False, 'learn_image_params':False, 'batch_size':1000, 'sort_method':'num_annos'}
SIMPLE_CROWDSOURCING = {'name':'majority-vote', 'line_style':'-.v', 'color':'m', 'use_computer_vision':False, 'online':False, 'simple_crowdsourcing':True, 'learn_worker_params':False, 'learn_image_params':False, 'batch_size':1000, 'sort_method':'num_annos'}

ALL_METHODS_NO_CV = [PROB_WORKER_IMAGE_ONLINE, PROB_ONLINE, PROB_WORKER_IMAGE, PROB_WORKER, PROB, SIMPLE_CROWDSOURCING]
ALL_PLOTS_NO_CV = [combine_dicts([{'title':'Method Comparison', 'name':'method_comparison_semilog', 'type':'semilog', 'legend':True, 'xlabel':'Avg Number of Human Workers Per Image', 'ylabel':'Error', 'methods':[{'name':PROB_WORKER_IMAGE_ONLINE['name'],'x':'num', 'y':'err'}, {'name':PROB_ONLINE['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER['name'],'x':'num', 'y':'err'}, {'name':PROB['name'],'x':'num', 'y':'err'}, {'name':SIMPLE_CROWDSOURCING['name'],'x':'num', 'y':'err'}]}, DEFAULT_PLOT_PARAMS]), 
                   combine_dicts([{'title':'Method Comparison', 'name':'method_comparison_loglog', 'type':'loglog', 'legend':True, 'xlabel':'Avg Number of Human Workers Per Image #', 'ylabel':'Error', 'methods':[{'name':PROB_WORKER_IMAGE_ONLINE['name'],'x':'num', 'y':'err'}, {'name':PROB_ONLINE['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER['name'],'x':'num', 'y':'err'}, {'name':PROB['name'],'x':'num', 'y':'err'}, {'name':SIMPLE_CROWDSOURCING['name'],'x':'num', 'y':'err'}]}, DEFAULT_PLOT_PARAMS]), 
                   combine_dicts([{'title':'Method Comparison', 'name':'method_comparison', 'type':'plot', 'legend':True, 'xlabel':'Avg Number of Human Workers Per Image', 'ylabel':'Error', 'methods':[{'name':PROB_WORKER_IMAGE_ONLINE['name'],'x':'num', 'y':'err'}, {'name':PROB_ONLINE['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER['name'],'x':'num', 'y':'err'}, {'name':PROB['name'],'x':'num', 'y':'err'}, {'name':SIMPLE_CROWDSOURCING['name'],'x':'num', 'y':'err'}]}, DEFAULT_PLOT_PARAMS]), 
                   combine_dicts([{'title':'Risk Estimation', 'name':'risk_estimation_semilog', 'type':'semilog', 'legend':True, 'xlabel':'Avg Number of Human Workers Per Image', 'ylabel':'Error', 'methods':[{'name':PROB_WORKER_IMAGE['name'], 'x':'num', 'y':'err', 'title':'Actual Error', 'line_style':'b-'}, {'name':PROB_WORKER_IMAGE['name'], 'x':'num', 'y':'risk', 'title':'Estimated Error', 'line_style':'g-'}]}, DEFAULT_PLOT_PARAMS]),
                   combine_dicts([{'title':'Risk Estimation', 'name':'risk_estimation_loglog', 'type':'loglog', 'legend':True, 'xlabel':'Avg Number of Human Workers Per Image', 'ylabel':'Error', 'methods':[{'name':PROB_WORKER_IMAGE['name'], 'x':'num', 'y':'err', 'title':'Actual Error', 'line_style':'b-'}, {'name':PROB_WORKER_IMAGE['name'], 'x':'num', 'y':'risk', 'title':'Estimated Error', 'line_style':'g-'}]}, DEFAULT_PLOT_PARAMS]),
                   combine_dicts([{'title':'Risk Estimation', 'name':'risk_estimation', 'type':'plot', 'legend':True, 'xlabel':'Avg Number of Human Workers Per Image', 'ylabel':'Error', 'methods':[{'name':PROB_WORKER_IMAGE['name'], 'x':'num', 'y':'err', 'title':'Actual Error', 'line_style':'b-'}, {'name':PROB_WORKER_IMAGE['name'], 'x':'num', 'y':'risk', 'title':'Estimated Error', 'line_style':'g-'}]}, DEFAULT_PLOT_PARAMS]),
                   combine_dicts([{'title':'Worker Skill', 'name':'worker_skill', 'type':'skill', 'methods':[{'name':PROB_WORKER_IMAGE['name']}]}, DEFAULT_PLOT_PARAMS]),
                   combine_dicts([{'title':'Num Annotations', 'name':'num_annotations', 'type':'hist', 'xlabel':'Annotations Per Image', 'ylabel':'Image Count', 'methods':[{'name':PROB_WORKER_IMAGE_ONLINE['name'], 'x':'num_annos', 'y':'num_annos_bins'}]}, DEFAULT_PLOT_PARAMS])
]

ALL_METHODS = [PROB_WORKER_IMAGE_CV_ONLINE_002, PROB_WORKER_IMAGE_CV_ONLINE_001, PROB_WORKER_IMAGE_CV_ONLINE_0005, PROB_WORKER_IMAGE_CV_NAIVE_ONLINE, PROB_WORKER_IMAGE_ONLINE, PROB_ONLINE, PROB_WORKER_IMAGE_CV, PROB_WORKER_IMAGE, PROB, SIMPLE_CROWDSOURCING]
ALL_PLOTS = [combine_dicts([{'title':'Method Comparison', 'name':'method_comparison_semilog', 'type':'semilog', 'legend':True, 'xlabel':'Avg Number of Human Workers Per Image', 'ylabel':'Error', 'methods':[{'name':PROB_WORKER_IMAGE_CV_ONLINE_002['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE_CV_ONLINE_0005['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE_CV_NAIVE_ONLINE['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE_ONLINE['name'],'x':'num', 'y':'err'}, {'name':PROB_ONLINE['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE_CV['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE['name'],'x':'num', 'y':'err'}, {'name':PROB['name'],'x':'num', 'y':'err'}, {'name':SIMPLE_CROWDSOURCING['name'],'x':'num', 'y':'err'}]}, DEFAULT_PLOT_PARAMS]), 
             combine_dicts([{'title':'Method Comparison', 'name':'method_comparison_loglog', 'type':'loglog', 'legend':True, 'xlabel':'Avg Number of Human Workers Per Image', 'ylabel':'Error', 'methods':[{'name':PROB_WORKER_IMAGE_CV_ONLINE_002['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE_CV_ONLINE_0005['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE_CV_NAIVE_ONLINE['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE_ONLINE['name'],'x':'num', 'y':'err'}, {'name':PROB_ONLINE['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE_CV['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE['name'],'x':'num', 'y':'err'}, {'name':PROB['name'],'x':'num', 'y':'err'}, {'name':SIMPLE_CROWDSOURCING['name'],'x':'num', 'y':'err'}]}, DEFAULT_PLOT_PARAMS]), 
             combine_dicts([{'title':'Method Comparison', 'name':'method_comparison', 'type':'plot', 'legend':True, 'xlabel':'Avg Number of Human Workers Per Image', 'ylabel':'Error', 'methods':[{'name':PROB_WORKER_IMAGE_CV_ONLINE_002['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE_CV_ONLINE_0005['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE_ONLINE['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE_CV_NAIVE_ONLINE['name'],'x':'num', 'y':'err'}, {'name':PROB_ONLINE['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE_CV['name'],'x':'num', 'y':'err'}, {'name':PROB_WORKER_IMAGE['name'],'x':'num', 'y':'err'}, {'name':PROB['name'],'x':'num', 'y':'err'}, {'name':SIMPLE_CROWDSOURCING['name'],'x':'num', 'y':'err'}]}, DEFAULT_PLOT_PARAMS]), 
             combine_dicts([{'title':'Risk Estimation', 'name':'risk_estimation_semilog', 'type':'semilog', 'legend':True, 'xlabel':'Avg Number of Human Workers Per Image', 'ylabel':'Error', 'methods':[{'name':PROB_WORKER_IMAGE_CV['name'], 'x':'num', 'y':'err', 'title':'Actual Error', 'line_style':'b-'}, {'name':PROB_WORKER_IMAGE_CV['name'], 'x':'num', 'y':'risk', 'title':'Estimated Error', 'line_style':'g-'}]}, DEFAULT_PLOT_PARAMS]),
             combine_dicts([{'title':'Risk Estimation', 'name':'risk_estimation_loglog', 'type':'loglog', 'legend':True, 'xlabel':'Avg Number of Human Workers Per Image', 'ylabel':'Error', 'methods':[{'name':PROB_WORKER_IMAGE_CV['name'], 'x':'num', 'y':'err', 'title':'Actual Error', 'line_style':'b-'}, {'name':PROB_WORKER_IMAGE_CV['name'], 'x':'num', 'y':'risk', 'title':'Estimated Error', 'line_style':'g-'}]}, DEFAULT_PLOT_PARAMS]),
             combine_dicts([{'title':'Risk Estimation', 'name':'risk_estimation', 'type':'plot', 'legend':True, 'xlabel':'Avg Number of Human Workers Per Image', 'ylabel':'Error', 'methods':[{'name':PROB_WORKER_IMAGE_CV['name'], 'x':'num', 'y':'err', 'title':'Actual Error', 'line_style':'b-'}, {'name':PROB_WORKER_IMAGE_CV['name'], 'x':'num', 'y':'risk', 'title':'Estimated Error', 'line_style':'g-'}]}, DEFAULT_PLOT_PARAMS]),
             combine_dicts([{'title':'Worker Skill', 'name':'worker_skill', 'type':'skill', 'methods':[{'name':PROB_WORKER_IMAGE['name']}]}, DEFAULT_PLOT_PARAMS]),
             combine_dicts([{'title':'Num Annotations', 'name':'num_annotations', 'type':'hist', 'xlabel':'Annotations Per Image', 'ylabel':'Image Count', 'methods':[{'name':PROB_WORKER_IMAGE_CV_ONLINE['name'], 'x':'num_annos', 'y':'num_annos_bins'}]}, DEFAULT_PLOT_PARAMS])
]
#-------------------------------------------------------



class SimulatedCrowdsourcer(object):
  def __init__(self, full_dataset, expert_dataset=None, save_prefix=None, output_dir='output', online=True, simple_crowdsourcing=False, learn_worker_params=True, learn_image_params=True, use_computer_vision=False, naive_computer_vision=False, batch_size=1000, num_rand_perms=1, sort_method='num_annos', name=None, line_style='-', color='r', save_all_perms=False, min_risk=0.005):
    self.full_dataset, self.expert_dataset = full_dataset, expert_dataset
    self.online, self.simple_crowdsourcing = online, simple_crowdsourcing
    self.learn_worker_params, self.learn_image_params = learn_worker_params, learn_image_params
    self.use_computer_vision, self.batch_size = use_computer_vision, (batch_size if online else len(full_dataset.images))
    self.naive_computer_vision = naive_computer_vision
    self.num_rand_perms, self.sort_method = num_rand_perms, sort_method
    self.save_prefix, self.output_dir = save_prefix, output_dir
    self.max_workers_per_image, self.save_all_perms = 1, save_all_perms
    self.min_risk = min_risk
    
  def run(self):
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)
    sum_plot_data, num_plot_data, fnames = [], [], []
    all_plot_data = {}
    for rand_perm in range(self.num_rand_perms):
      plot_data = {'num':[], 'err':[], 'risk':[]}
      iter = 0
      self.rand_perms = {}
      #if self.full_dataset.computer_vision_predictor and hasattr(self.full_dataset.computer_vision_predictor, 'iteration'):
      #  self.full_dataset.computer_vision_predictor.iteration = 0
      self.dataset = self.full_dataset.__class__(debug=0, learn_worker_params=self.learn_worker_params, learn_image_params=self.learn_image_params, computer_vision_predictor=(self.full_dataset.computer_vision_predictor if self.use_computer_vision else None), naive_computer_vision=self.naive_computer_vision, min_risk=self.min_risk)
      self.dataset.copy_parameters_from(self.full_dataset, full=False)
      for i in self.full_dataset.images: 
        self.dataset.images[i] = self.full_dataset._CrowdImageClass_(i, self.dataset)
        self.dataset.images[i].copy_parameters_from(self.full_dataset.images[i], full=False)
      for w in self.full_dataset.workers: 
        self.dataset.workers[w] = self.full_dataset._CrowdWorkerClass_(w, self.dataset)
        self.dataset.workers[w].copy_parameters_from(self.full_dataset.workers[w], full=False)
      
      while self.dataset.num_unfinished(full_dataset=self.full_dataset) > 0:
        if self.simple_crowdsourcing:
          self.dataset.crowdsource_simple()
        else:
          self.dataset.estimate_parameters(avoid_if_finished=True)
          self.dataset.check_finished_annotations(set_finished=self.online)
        if self.expert_dataset:
          err,num = self.dataset.compute_error(self.expert_dataset), self.dataset.num_annotations()
          if not self.simple_crowdsourcing: 
            plot_data["risk"].append(self.dataset.risk())
          plot_data["num"].append(float(num)/len(self.dataset.images))
          plot_data["err"].append(err)
        if self.save_prefix and rand_perm==0 or self.save_all_perms:
          fname = self.output_dir+'/'+self.save_prefix+str(rand_perm)+'_'+str(iter)+'.json'
          self.dataset.save(fname)
          fnames.append({str(rand_perm)+'_'+str(iter):fname})
        self.augment_annotations_if_necessary()        
        iter += 1
      if hasattr(self.dataset, 'parts'):
        plot_data["num_annos"] = []
        for p in range(len(self.dataset.parts)):
          plot_data["num_annos"] += [self.dataset.parts[p].images[i].num_annotations() for i in self.dataset.parts[p].images]
      else:
        plot_data["num_annos"] = [self.dataset.images[i].num_annotations() for i in self.dataset.images]
      plot_data["num_annos_bins"] = np.arange(-.5, np.asarray(plot_data["num_annos"]).max()+.5, 1).tolist()
      if hasattr(self.dataset.workers[w],'skill') and self.dataset.workers[w].skill:
        for s in range(len(self.dataset.workers[self.dataset.workers.keys()[0]].skill)): 
          plot_data["skill"+str(s)] = [self.dataset.workers[w].skill[s] for w in self.dataset.workers]
          if self.dataset.cv_worker: plot_data["skill_cv"+str(s)] = [self.dataset.cv_worker.skill[s]]
      plot_data["worker_num_annos"] = [len(self.dataset.workers[w].images) for w in self.dataset.workers]
      for k in plot_data: 
        if not k in all_plot_data:
          all_plot_data[k] = []
        all_plot_data[k].append(plot_data[k])
      
    plot_data = {}
    for k in all_plot_data:
      ml = int(np.asarray([len(c) for c in all_plot_data[k]]).max())
      a = np.zeros((self.num_rand_perms, ml)) 
      valid = np.zeros((self.num_rand_perms, ml)) 
      for i in range(self.num_rand_perms):
        a[i,:len(all_plot_data[k][i])] = all_plot_data[k][i]
        valid[i,:len(all_plot_data[k][i])] = 1
      if k == 'num_annos':
        plot_data[k] = a.flatten().tolist()
      else:
        plot_data[k] = (a.sum(axis=0) / valid.sum(axis=0)).tolist()
        plot_data[k+'_var'] = ((((a-plot_data[k])**2)*valid).sum(axis=0) / valid.sum(axis=0)).tolist()
      
    return plot_data, fnames, all_plot_data
    
  def augment_annotations_if_necessary(self):
    processed = []
    num = 0
    image_ids = self.dataset.choose_images_to_annotate_next(max_workers_per_image=self.max_workers_per_image, sort_method=self.sort_method, full_dataset=self.full_dataset)
    for i in image_ids:
      processed.append(i)
      workers = self.full_dataset.images[i].workers
      if not i in self.rand_perms:
        self.rand_perms[i] = np.random.permutation(len(self.full_dataset.images[i].z))
      #print str(i) + " " + str(len(workers)) + " " + str(workers)
      has_cv = (1 if (self.dataset.cv_worker and self.dataset.cv_worker.id in self.dataset.images[i].z) else 0)
      fd_has_cv = (1 if (self.full_dataset.cv_worker and self.full_dataset.cv_worker.id in self.full_dataset.images[i].z) else 0)
      for j in range(len(self.dataset.images[i].z)-has_cv, min(len(self.dataset.images[i].z)-has_cv+self.max_workers_per_image, len(self.full_dataset.images[i].z)-fd_has_cv)):
        w = workers[self.rand_perms[i][j]]
        if not self.dataset.images[i].finished:
          assert not w in self.dataset.images[i].z, "Duplicate worker " + str(w) + " for image " + str(i) + " calling augment_annotations()"
          z = self.dataset._CrowdLabelClass_(self.dataset.images[i], self.dataset.workers[w])
          z.parse(self.full_dataset.images[i].z[w].raw_data)
          self.dataset.images[i].z[w] = z
          self.dataset.images[i].workers.append(w)
          self.dataset.workers[w].images[i] = self.dataset.images[i]
          num += 1
      if num >= self.batch_size: 
        break
    '''
    if self.use_computer_vision: 
      for i in processed:   
        self.dataset.images[i].predict_true_labels(avoid_if_finished=True)  # Initialize training label for computer vision
      self.dataset.synch_computer_vision_labels()  
    '''
  


def RunSimulatedExperiments(full_dataset, methods, output_dir, plots, expert_dataset=None, title=None, num_rand_perms=5, force_compute=False, show_intermediate_results=True):
  results, fnames, methods_d, all_plot_data = {}, {}, {}, {}
  fnames_i = []
  for a in methods:
    m = a['name']
    if not os.path.exists(os.path.join('output', output_dir, 'plot_data_'+m+'.json')) or force_compute:
      sc = SimulatedCrowdsourcer(full_dataset, expert_dataset=(expert_dataset if expert_dataset else full_dataset), num_rand_perms=num_rand_perms, output_dir='output/'+output_dir, save_prefix=m, **a)
      results[m], fnames[m], all_plot_data[m] = sc.run()
      with open(os.path.join('output', output_dir, 'plot_data_'+m+'.json'), 'w') as f:
        json.dump({'results':results[m], 'fnames':fnames[m], 'all_plot_data':all_plot_data[m]}, f)
    else:
      with open(os.path.join('output', output_dir, 'plot_data_'+m+'.json')) as f:
        data = json.load(f)
        results[m], fnames[m], all_plot_data[m] = data['results'], data['fnames'], data['all_plot_data']
    methods_d[m] = a
    fnames_c = []
    for f in fnames[a['name']]:
      fnames_c.append({f.keys()[0] : f[f.keys()[0]][len('output/'):]})
    fnames_i.append({'name':a['name'], 'files':fnames_c})
    if show_intermediate_results: 
      GeneratePlotResults(full_dataset, methods, output_dir, plots, results, methods_d, fnames_i, title)
    
  GeneratePlotResults(full_dataset, methods, output_dir, plots, results, methods_d, fnames_i, title)

def GeneratePlotResults(full_dataset, methods, output_dir, plots, results, methods_d, fnames_i, title):
  plot_files = []
  for i in range(len(plots)):
    handles, labels = [], []
    fig = plt.figure(i+1)
    plt.clf()
    plot = plots[i]
    if 'xlim' in plot: plt.xlim(plot['xlim'][0], plot['xlim'][1])
    if 'ylim' in plot: plt.ylim(plot['ylim'][0], plot['ylim'][1])
    for a in plots[i]['methods']:
      m = a['name']
      if not m in methods_d:
        continue
      print str(i) + ' ' + str(m)
      line_style = a['line_style'] if 'line_style' in a else methods_d[m]['line_style']
      color = a['color'] if 'color' in a else methods_d[m]['color']
      if plot['type'] == 'skill':
         plot = copy.deepcopy(plots[i])
         plot['type'] = 'scatter' if len(full_dataset.skill_names) <= 2 else 'scatter3d'
         if not 'xlabel' in plot: plot['xlabel'] = full_dataset.skill_names[0]
         if not 'ylabel' in plot: plot['ylabel'] = full_dataset.skill_names[1]
         if not 'zlabel' in plot and len(full_dataset.skill_names) > 2: plot['zlabel'] = full_dataset.skill_names[2]
         if not 'x' in a: a['x'] = 'skill0'
         if not 'y' in a: a['y'] = 'skill1'
         if not 'z' in a and len(full_dataset.skill_names) > 2: a['z'] = 'skill2'
      if plot['type'] == 'semilog':
        print str(len(results[m][a['x']])) + ' ' + str(len(results[m][a['y']]))
        h = plt.semilogy(results[m][a['x']], results[m][a['y']], line_style, lw=plot["line-width"], color=color)
        handles.append(h[0])
      elif plot['type'] == 'loglog':
        h = plt.loglog(results[m][a['x']], results[m][a['y']], line_style, lw=plot["line-width"], color=color)
        handles.append(h[0])
      elif plot['type'] == 'plot':
        h = plt.plot(results[m][a['x']], results[m][a['y']], line_style, lw=plot["line-width"])
        handles.append(h[0])
      elif plot['type'] == 'hist':
        if 'y' in a:
          h = plt.hist(results[m][a['x']], results[m][a['y']], histtype='bar', rwidth=plot["bar-width"], color=plot["bar-color"])
        else:
          h = plt.hist(results[m][a['x']], histtype='bar', rwidth=plot["bar-width"], color=plot["bar-color"]) 
        handles.append(h[0])
      elif plot['type'] == 'scatter':
        h = plt.scatter(results[m][a['x']], results[m][a['y']], c='r', marker='o')   
        handles.append(h)
      elif plot['type'] == 'scatter3d':
        ax = fig.add_subplot(111, projection='3d')
        h = ax.scatter(results[m][a['x']], results[m][a['y']], results[m][a['z']], c='r', marker='o')
        handles.append(h)
      labels.append(a['title'] if 'title' in a else m)
    if plot['type'] == 'scatter3d':
      if 'xlabel' in plot: ax.set_xlabel(plot['xlabel'], fontsize=plot['axis-font-size'])
      if 'ylabel' in plot: ax.set_ylabel(plot['ylabel'], fontsize=plot['axis-font-size'])
      if 'zlabel' in plot: ax.set_zlabel(plot['zlabel'], fontsize=plot['axis-font-size'])
    else:
      if 'xlabel' in plot: plt.xlabel(plot['xlabel'], fontsize=plot['axis-font-size'])
      if 'ylabel' in plot: plt.ylabel(plot['ylabel'], fontsize=plot['axis-font-size'])
      if 'zlabel' in plot: plt.zlabel(plot['zlabel'], fontsize=plot['axis-font-size'])
    if 'title' in plot: plt.title(plot['title'], fontsize=plot['title-font-size'])
    plt.tick_params(axis='both', which='major', labelsize=plot['tick-font-size'])
    plt.tick_params(axis='both', which='minor', labelsize=plot['tick-font-size'])
    if 'legend' in plot: plt.legend(handles, labels, prop={'size':plot['legend-font-size']})
    plt.savefig(os.path.join('output', output_dir, plot['name']+'.pdf'))
    plt.savefig(os.path.join('output', output_dir, plot['name']+'.png'))
    plot_files.append(output_dir + '/' + plot['name']+'.png')
    
  with open(os.path.join('output', output_dir, 'galleries.json'), 'w') as f:
    json.dump({'plots':plot_files, 'methods':fnames_i, 'title':title}, f)
