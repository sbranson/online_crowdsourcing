from boto.mturk.connection import MTurkConnection
from boto.mturk.question import ExternalQuestion
from boto.mturk.qualification import *
from boto.mturk.price import Price
from boto.resultset import ResultSet
from PIL import Image
import os
import BaseHTTPServer, SimpleHTTPServer
import CGIHTTPServer
import ssl
import copy
import math
import json
import time
import threading
import posixpath
import argparse
import urllib
import pickle
import sys

class MTurkCrowdsourcer(object):
  def __init__(self, dataset, aws_access_key, aws_secret_access_key, host, output_folder, output_web_folder = 'online_crowdsourcing', 
               online = True, initial_assignments_per_image = 0, batch_size = 1000, port=443, python_webserver=False, ssl_certfile=None,
               max_annos=float('Inf'), hit_params = None, thumbnail_size = None, description=None, reward=None, keywords=None, 
               images_per_hit=None, title=None, duration=3600, approval_delay=86400, qualifications=None, sandbox=True
           ):
    self.dataset = dataset
    self.online = online
    self.initial_assignments_per_image = initial_assignments_per_image
    self.batch_size = batch_size
    self.hit_params = hit_params if not hit_params is None else dataset.hit_params
    self.thumbnail_size = thumbnail_size
    self.max_annos = max_annos

    self.ssl_certfile = ssl_certfile
    self.python_webserver = python_webserver
    self.images_per_hit = dataset.images_per_hit if images_per_hit is None else images_per_hit
    self.host = host
    self.output_folder = output_folder
    self.output_dir = 'output/' + output_folder
    self.port = port
    self.base_url =  'https://' + host + ((':' + str(port)) if (port and port != 443) else '') + '/' + output_web_folder + '/' + self.output_folder
    if not hasattr(self.dataset, 'fname'): 
      self.dataset.fname = os.path.join(self.output_dir, 'dataset.json')
    
    self.account = AWSAccount(aws_access_key, aws_secret_access_key)
    self.hit_type = HitType(self.account, 
                            title = dataset.title if title is None else title,
                            description = dataset.description if description is None else description,
                            reward = dataset.reward if reward is None else reward,
                            keywords = dataset.keywords if keywords is None else keywords,
                            duration = duration,approval_delay = approval_delay, 
                            qualifications = qualifications, sandbox = sandbox)
    self.batch_num = 0
    self.max_assignments = self.initial_assignments_per_image

  def run(self, resume_hits = None):
    print "Initializing images/thumbnails for webpages..."
    self.initialize_web_directories()
    
    if self.hit_type.status != 'Registered':
      self.hit_type.register()
    
    if self.python_webserver:
      print "start webserver"
      webserver_thread = threading.Thread(target=self.run_webserver)
      webserver_thread.start()

    while (self.online and self.dataset.num_unfinished(max_annos=self.max_annos) > 0) or self.batch_num==0:
      # Select unfinished images
      print "Choosing images to annotate..."
      image_ids = self.dataset.choose_images_to_annotate_next()
      n = len(image_ids)
      if self.batch_num > 0 or self.initial_assignments_per_image == 0:
        n = min(self.batch_size, n) 
        self.max_assignments = 1
      
      # Create HITs on Mechanical Turk
      if resume_hits is None:
        num_hits = math.ceil(n / float(self.images_per_hit))
        print "Creating " + str(int(num_hits)) + " HITs..."
        hits = []
        images_per_hit = int(math.ceil(float(n) / num_hits))
        for i in range(0, n, images_per_hit):
          hits.append(self.create_hit(image_ids[i:min(i+images_per_hit,n)], str(self.batch_num) + "_" + str(len(hits)), max_assignments=self.max_assignments))
      
        sys.setrecursionlimit(10000)
        with open(os.path.join(self.output_dir, 'batch' + str(self.batch_num) + '.pkl'), 'wb') as f:
          pickle.dump({'crowdsourcer':self, 'hits':hits}, f)
      else:
        hits = resume_hits
        resume_hits = None
      
      # Wait for results to come back from MTurk
      print "Waiting for workers to finish HITs..."
      self.wait_for_hit_results(hits)
      
      # Predict image annotations and check which images are finished
      print "Predicting image labels..."
      self.dataset.estimate_parameters(avoid_if_finished=True)
      self.dataset.check_finished_annotations()
      
      with open(os.path.join(self.output_dir, 'batch_results' + str(self.batch_num) + '.pkl'), 'wb') as f:
        pickle.dump({'crowdsourcer':self, 'hits':hits}, f)
      self.save_dataset_visualization()
      
      self.batch_num += 1
  
  def initialize_web_directories(self):
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)
    if not os.path.exists(self.output_dir + '/images'):
      os.makedirs(self.output_dir + '/images')
    if not os.path.exists(self.output_dir + '/cgi-bin'):
      os.makedirs(self.output_dir + '/cgi-bin')
    if (not self.thumbnail_size is None) and not os.path.exists(self.output_dir + '/thumbs'):
      os.makedirs(self.output_dir + '/thumbs')
    if not os.path.exists(self.output_dir + '/hits'):
      os.makedirs(self.output_dir + '/hits')
    for i in self.dataset.images:
      path, ext = os.path.splitext(self.dataset.images[i].fname)
      if not hasattr(self.dataset.images[i], 'url'):
        self.dataset.images[i].url = self.base_url + '/images/' + str(i) + ext
        if not os.path.exists(self.output_dir + '/images/' + str(i) + ext):
          out_dir = self.output_dir + '/images/'
          os.symlink(os.path.relpath(self.dataset.images[i].fname, out_dir), out_dir + str(i) + ext)
      if (not self.thumbnail_size is None):
        if not os.path.exists(self.output_dir + '/thumbs/' + str(i) + ext):
          img = Image.open(self.dataset.images[i].fname)
          d = img.size[1]-img.size[0]
          if d > 0:
            img = img.crop((0, d/2, img.size[0], img.size[1]-d/2))
          elif d < 0:
            img = img.crop((-d/2, 0, img.size[0]+d/2, img.size[1]))
          img.thumbnail(self.thumbnail_size, Image.ANTIALIAS)
          img.save(self.output_dir + '/thumbs/' + str(i) + ext)
        self.dataset.images[i].thumb_url = self.base_url + '/thumbs/' + str(i) + ext
      else:
        self.dataset.images[i].thumb_url = self.dataset.images[i].url
    if not os.path.exists(self.output_dir + '/hits/js'):
      os.symlink(os.path.relpath('html/js', self.output_dir + '/hits'), self.output_dir + '/hits/js')
    
  def run_webserver(self):
    print "Run webserver " + self.host + " " + self.output_dir
    handler = RootedHTTPRequestHandler
    handler.base_path = self.output_dir
    handler.cgi_directories = ['/cgi-bin']
    httpd = BaseHTTPServer.HTTPServer((self.host, self.port), handler)
    if self.ssl_certfile:
      httpd.socket = ssl.wrap_socket (httpd.socket, certfile=self.ssl_certfile, server_side=True)
    httpd.serve_forever()

  def create_hit(self, image_ids, hit_id, max_assignments=1):
    hit = Hit(self.hit_type, self.build_hit_params(image_ids), 
              self.output_dir + "/hits/" + str(hit_id) + ".html", 
              self.base_url + "/hits/" + str(hit_id) + ".html", 
              max_assignments=max_assignments)
    replace_in_file(self.dataset.html_template_dir+"/index.html", hit.html_file, 
                    {"<!-- Insert HIT Parameters Here -->" : ("gParameters = " + json.dumps(hit.params) +";") })
    hit.register()
    return hit
  
  def build_hit_params(self, image_ids):
    params = copy.copy(self.hit_params)
    params['image_ids'] = image_ids
    params['is_mturk_hit'] = True
    params['image_urls'], params['image_thumb_urls'] = {}, {}
    for i in image_ids:
      params['image_urls'][i] = self.dataset.images[i].url
      params['image_thumb_urls'][i] = self.dataset.images[i].thumb_url
    return params

  def parse_response(self, raw_data, a, i):
    if not a.WorkerId in self.dataset.workers:
      self.dataset.workers[a.WorkerId] = self.dataset._CrowdWorkerClass_(a.WorkerId, self.dataset)
    z = self.dataset._CrowdLabelClass_(self.dataset.images[i], self.dataset.workers[a.WorkerId])
    z.parse(raw_data)
    self.dataset.images[i].z[a.WorkerId] = z
    self.dataset.images[i].workers.append(a.WorkerId)
    self.dataset.workers[a.WorkerId].images[i] = self.dataset.images[i]
  
  def wait_for_hit_results(self, hits):
    unfinished_hits = hits[:]
    self.hits = hits
    while(len(unfinished_hits)):
      new_unfinished = []
      foundNew = False
      sleepTime = 1
      for h in unfinished_hits:
        #print "Hit " + str(h.url) + " " + str(len(h.assignments)) + '/' + str(h.max_assignments) + " assignments completed"
        if not h.get_completed_assignments():
          new_unfinished.append(h)
        for j in range(h.num_parsed_assignments, len(h.assignments)):
          print "Hit " + str(h.url) + " parsing new assignment"
          foundNew = True
          sleepTime = 1
          a = h.assignments[j]
          if 'json' in a.raw_data:
            raw_data = json.loads(a.raw_data["json"])
            if "annos" in raw_data:
              for r in raw_data["annos"]:
                self.parse_response(r['anno'], a, r['image_id'])
          else:
            for k in a.raw_data:
              i = k[3:] if k.startswith('img') else (k[5:] if k.startswith('image') else None)
              if not i is None:
                raw_data = a.raw_data[k] if type(a.raw_data[k]) == dict else {'label':a.raw_data[k]}
                self.parse_response(raw_data, a, i)
          h.num_parsed_assignments = len(h.assignments)
      if len(new_unfinished) and not foundNew:
        self.save_dataset_visualization(fname = 'in_progress_dataset.json')
        time.sleep(sleepTime)
        sleepTime = min(60, sleepTime*2)
      unfinished_hits = new_unfinished

  def save_dataset_visualization(self, fname=None):
    galleries = '['
    for i in range(self.batch_num):
      if i > 0: galleries += ','
      galleries += '["' + ('batch' + str(i) + '_dataset.json') + '","' + ('batch' + str(i)) + '"]'
    if not fname is None:
      if self.batch_num > 0: galleries += ','
      galleries += '["'+fname+'","In Progress"]'
    else:
      fname = 'batch' + str(self.batch_num) + '_dataset.json'
    galleries += ']'
    
    params = {'<<galleries>>':galleries}
    replace_in_file('output/html/monitor.html', os.path.join(self.output_dir,'monitor.html'), params)
    self.dataset.save(os.path.join(self.output_dir, fname))
    
def ResumeMTurk(output_dir):
  i = 0
  while os.path.exists(os.path.join(output_dir, 'batch'+str(i)+'.pkl')):
    i += 1
  if i > 0:
    with open(os.path.join(output_dir, 'batch'+str(i-1)+'.pkl')) as f: 
      data = pickle.load(f)
      crowdsource = data['crowdsourcer']
      crowdsource.run(resume_hits = data['hits'])
      return crowdsource

class AWSAccount(object):
  def __init__(self, aws_access_key, aws_secret_access_key):
    self.aws_secret_access_key = aws_secret_access_key
    self.aws_access_key = aws_access_key
    self.connections = {}
  
  def connection(self, sandbox):
    url = 'mechanicalturk.sandbox.amazonaws.com' if sandbox else 'mechanicalturk.amazonaws.com'
    if not url in self.connections:
      self.connections[url] = MTurkConnection(aws_access_key_id=self.aws_access_key,
                                              aws_secret_access_key=self.aws_secret_access_key, host=url)
    return self.connections[url]
  
  def get_balance(self):
    mtc = self.connection(False)
    rs = mtc.get_account_balance()
    if not rs.status:
      return False
    return rs[0]

class HitType(object):
  def __init__(self, account, title="", description="", reward=0.01, duration=3600, keywords="", 
               approval_delay=86400, qualifications=None, sandbox=False):
    self.account = account
    self.description = description
    self.reward = reward
    self.duration = duration
    self.keywords = keywords
    self.approval_delay = approval_delay
    self.qualifications = qualifications
    self.sandbox = sandbox
    self.mturk_id = None
    self.title = title
    self.status = 'Created'

  def register(self):
    mtc = self.account.connection(self.sandbox)
    rs = mtc.register_hit_type(title=self.title, description=self.description,
                               reward=Price(self.reward), duration=self.duration,
                               keywords=self.keywords,
                               approval_delay=self.approval_delay,
                               qual_req=build_qualifications(self.qualifications))
    if rs.status:
      self.status = 'Registered'
      self.mturk_id = rs[0].HITTypeId
      return True
    return False
  
class Hit(object):
  def __init__(self, hit_type, params, html_file, url, max_assignments=1, lifetime=259200, frame_height=800):
    self.hit_type = hit_type
    self.account = hit_type.account
    self.params = params
    self.html_file = html_file
    self.url = url
    self.max_assignments = max_assignments
    self.lifetime = lifetime
    self.frame_height = frame_height
    self.mturk_id = None
    self.assignments = []
    self.num_parsed_assignments = 0
  
  def register(self):
    if self.account is None: return False
    mtc = self.account.connection(self.hit_type.sandbox)
    rs = mtc.create_hit(hit_type=self.hit_type.mturk_id,
                        question=ExternalQuestion(external_url=self.url,
                                                  frame_height=self.frame_height),
                        lifetime=self.lifetime, max_assignments=self.max_assignments)
    if rs.status:
      self.mturk_id = rs[0].HITId
      self.status = 'Unassigned'
      return True
    return False
  
  def get_completed_assignments(self):
    if self.hit_type.account is None or self.mturk_id is None: 
      return False
    mtc = self.hit_type.account.connection(self.hit_type.sandbox)
    if len(self.assignments) < self.max_assignments and not self.status=='Disposed':
      rs = mtc.get_assignments(hit_id=self.mturk_id, page_size=self.max_assignments)
      if rs.status:
        self.assignments = []
        for ra in rs:
          ra.raw_data = {}
          for b in ra.answers[0]:
            ra.raw_data[b.qid] = b.fields[0]
          self.assignments.append(ra)
      return len(self.assignments) >= self.max_assignments
    return True
  
  def disable(self):
    if self.hit_type.account is None: return False
    if self.mturk_id is None or self.status=='Disposed':
      return False
    mtc = self.account.connection(self.hit_type.sandbox)
    rs = mtc.disable_hit(hit_id=self.mturk_id)
    if rs.status:
      self.status = 'Disposed'
    return rs.status
    
  def disable(self):
    if self.hit_type.account is None: return False
    if self.mturk_id is None or self.status=='Reviewable':
      return False
    mtc = self.account.connection(self.hit_type.sandbox)
    rs = mtc.dispose_hit(hit_id=self.mturk_id)
    if rs.status:
      self.status = 'Disposed'
    return rs.status
    
  def expire(self):
    if self.hit_type.account is None: return False
    if self.mturk_id is None:
      return False
    mtc = self.account.connection(self.hit_type.sandbox)
    rs = mtc.expire_hit(hit_id=self.mturk_id)
    return rs.status

  def extend(self, assignments_increment=None, expiration_increment=None):
    if self.hit_type.account is None: return False
    if self.mturk_id is None:
      return False
    mtc = self.account.connection(self.hit_type.sandbox)
    rs = mtc.extend_hit(hit_id=self.mturk_id, assignments_increment=assignments_increment,
                        expiration_increment=expiration_increment)
    return rs.status

  def set_reviewing(self, revert=False):
    if self.account is None: return False
    req_status = self.status = 'Reviewing' if revert else 'Reviewable'
    if self.mturk_id is None or not self.status==req_status:
      return False
    mtc = self.account.connection(self.hit_type.sandbox)
    rs = mtc.set_reviewing(hit_id=self.mturk_id, revert=revert)
    if rs.status:
      self.status = 'Reviewable' if revert else 'Reviewing'
    return rs.status

def build_qualifications(qualifications):
  if qualifications is None or qualifications=='':
    return None
  qs = json.loads(self.qualifications)
  qr = Qualifications()
  for q in qs:
    if q['qualification_type'] == "Adult":
      qr.add(AdultRequirement(q['comparator'], q['value'],
                              q['required_to_preview']))
    elif q['qualification_type'] == "Locale":
      qr.add(LocaleRequirement(q['comparator'], q['value'],
                               q['required_to_preview']))
    elif q['qualification_type'] == "NumberHitsApproved":
      qr.add(NumberHitsApprovedRequirement(q['comparator'], q['value'],
                                           q['required_to_preview']))
    elif q['qualification_type'] == "PercentAssignmentsAbandoned":
      qr.add(PercentAssignmentsAbandonedRequirement(q['comparator'], q['value'],
                                                    q['required_to_preview']))
    elif q['qualification_type'] == "PercentAssignmentsApproved":
      qr.add(PercentAssignmentsApprovedRequirement(q['comparator'], q['value'],
                                                   q['required_to_preview']))
    elif q['qualification_type'] == "PercentAssignmentsRejected":
      qr.add(PercentAssignmentsRejectedRequirement(q['comparator'], q['value'],
                                                   q['required_to_preview']))
    elif q['qualification_type'] == "PercentAssignmentsReturned":
      qr.add(PercentAssignmentsReturnedRequirement(q['comparator'], q['value'],
                                                   q['required_to_preview']))
    elif q['qualification_type'] == "PercentAssignmentsSubmitted":
      qr.add(PercentAssignmentsSubmittedRequirement(q['comparator'], q['value'],
                                                    q['required_to_preview']))
    else:
      qr.add(Requirement(Qualification.objects.get(pk=q['qualification_type']).mturk_id,
                         q['comparator'], q['value'], q['required_to_preview']))
  return qr


def replace_in_file(src_file, dst_file, match_replace_dict):
  with open (src_file, "r") as f:
    s = f.read()
    for match,replace in match_replace_dict.iteritems():
      s = s.replace(str(match), str(replace))
  if not dst_file is None:
    with open(dst_file, 'w') as f:
      f.write(s)
  return s

class RootedHTTPRequestHandler(CGIHTTPServer.CGIHTTPRequestHandler):
  def translate_path(self, orig_path):
    path = self.base_path
    for word in filter(None, posixpath.normpath(urllib.unquote(orig_path.split('?',1)[0].split('#',1)[0])).split('/')):
      drive, word = os.path.splitdrive(word)
      head, word = os.path.split(word)
      if word in (os.curdir, os.pardir):
        continue
      path = os.path.join(path, word)
    return path
