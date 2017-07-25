from mechanical_turk import *
from boto.resultset import ResultSet
from boto.mturk.connection import QuestionFormAnswer
import os
import BaseHTTPServer, SimpleHTTPServer
import json
import time
import CGIHTTPServer
import cgitb; cgitb.enable() 
import urlparse
import urllib2
import re
import numpy as np
import requests


class LocalCrowdsourcer(MTurkCrowdsourcer):
  def __init__(self, dataset, host, output_folder, **kwds):
    super(LocalCrowdsourcer, self).__init__(dataset, None, None, host, output_folder, python_webserver=True, output_web_folder = '', **kwds)
    self.base_url =  'http://' + host + ((':' + str(self.port)) if (self.port and self.port != 80) else '') 
    self.account = SpoofedAWSAccount(self.output_dir, self.base_url)
    self.hit_type.account = self.account

  def initialize_web_directories(self):
    super(LocalCrowdsourcer, self).initialize_web_directories()
    if not os.path.exists(os.path.join(self.output_dir, 'visualize')):
      os.symlink('../html', os.path.join(self.output_dir, 'visualize'))
    print "Annotation url is " + self.base_url + "/cgi-bin/annotate0.py"
    print "Visualize annotation progress at " + self.base_url + "/monitor.html"
    #if not os.path.exists(os.path.join(self.output_dir, 'cgi-bin')):
    #  os.makedirs(os.path.join(self.output_dir, 'cgi-bin'))

  def build_hit_params(self, image_ids):
    params = super(LocalCrowdsourcer, self).build_hit_params(image_ids)
    params['is_mturk_hit'] = False
    return params

class SpoofedAWSAccount(object):
  def __init__(self, output_dir, url):
    self.output_dir = output_dir
    self.url = url
  
  def connection(self, sandbox):
    return SpoofedMTurkConnection(self.output_dir, self.url)

class HitTypeResponse(object):
  def __init__(self, **kwargs):
    for name, value in kwargs.items():
      setattr(self, name, value)

class HitResponse(object):
  def __init__(self, **kwargs):
    for name, value in kwargs.items():
      setattr(self, name, value)

class Assignment(object):
  def __init__(self, **kwargs):
    for name, value in kwargs.items():
      setattr(self, name, value)

class SpoofedMTurkConnection(object):
  def __init__(self, output_dir, url):
    self.output_dir = output_dir
    self.url = url
    self.num_hit_types = 0
  
  def register_hit_type(self, title='', description='', reward=Price(0.01), duration=3600, keywords='', approval_delay=3600*24, qual_req=None):
    hit_type_id = self.num_hit_types
    self.num_hit_types += 1
    if not os.path.exists(self.output_dir): 
      os.makedirs(self.output_dir)
    with open(os.path.join(self.output_dir, "hit_type"+str(hit_type_id)+".json"), 'w') as f:
      json.dump({'hits':[], 'finished':False, 'duration':duration, 'approval_delay':approval_delay, 'keywords':keywords, 
                 'reward':reward.amount, 'title':title, 'description':description, 'workers':{}}, f)
    os.chmod(os.path.join(self.output_dir, "hit_type"+str(hit_type_id)+".json"), 0666)
    rs = ResultSet()
    rs.append(HitTypeResponse(HITTypeId=hit_type_id))
  
    with open(os.path.join(self.output_dir, 'cgi-bin', 'new_assignment.py'), 'w') as f: 
      f.write('''#!/usr/bin/env python
import os
import datetime
import json
import time
import numpy as np

def hit_add_assignment(h, hit_type, hit_type_id, curr, output_dir, worker_id, submit_url):
  found = False
  for w in h[3]:
    if w == worker_id:
      found = True
  if not found:
    h[2] += 1
    h[3].append(worker_id)
    with open(os.path.join(output_dir, "hit_type"+str(hit_type_id)+".json"), 'w') as f:
      json.dump(hit_type, f)
    with open(os.path.join(output_dir, "hit"+str(h[0])+".json"), 'r') as f:
      hit = json.load(f)
    params = "?hitTypeId="+str(hit_type_id)+"&hitId="+str(hit['mturk_id'])+"&workerId="+str(worker_id) + "&turkSubmitTo=" + submit_url
    a = {'accept_time':curr, 'worker':worker_id, 'status':'In Progress', 'hit_id':hit['mturk_id'], 'external_url':hit['external_url']+params}
    hit['assignments'].append(a)
    with open(os.path.join(output_dir, "hit"+str(h[0])+".json"), 'w') as f:
      json.dump(hit, f)
    return a

def new_assignment(output_dir, worker_id, hit_type_id):
  submit_url = "''' + self.url + '''/cgi-bin/submit"+str(hit_type_id)+".py"
  curr = time.time()
  with open(os.path.join(output_dir, "hit_type"+str(hit_type_id)+".json"), 'r') as f:
    hit_type = json.load(f)
  if hit_type['finished']: 
    return None
  
  # Choose a random, unfinished hit that this worker hasn't done already
  for hi in np.random.permutation(len(hit_type['hits'])):
    h = hit_type['hits'][hi]
    if h[2] < h[1]:
      hit_add_assignment(h, hit_type, hit_type_id, curr, output_dir, worker_id, submit_url)
  
  # Handle expired hits and assignments
  new_hit_available = None
  any_changed = False
  in_progress_assignment, in_progress_hit, almost_finished_hit = None, None, None
  for h in hit_type['hits']:
    changed = False
    if h[2] >= h[1] and not h[4]:
      with open(os.path.join(output_dir, "hit"+str(h[0])+".json"), 'r') as f:
        hit = json.load(f)
      if curr >= hit['create_time'] + hit['lifetime']:
        hit['status'] = 'Expired'
        h[4] = 1
        changed = True
      num_finished = 0
      num_in_progress = 0
      for a in hit['assignments']:
        if curr > a['accept_time']+hit_type['duration'] and a['status']=='In Progress':  # Expired assignment
          a['status'] = 'Expired'
          h[2] -= 1
          changed = True
        elif a['worker'] == worker_id and a['status']=='In Progress':
          in_progress_assignment = a
          in_progress_hit = hit
        elif a['status']=='In Progress':
          num_in_progress += 1
        if a['status']=='Submitted':
          num_finished += 1
      if num_finished >= h[1]:
        h[4] = 1
        changed = True
      elif num_in_progress:
        almost_finished_hit = h
      if changed:
        any_changed = True
        with open(os.path.join(output_dir, "hit"+str(h[0])+".json"), 'w') as f:
          json.dump(hit, f)
        if h[2] < h[1]:
          new_hit_available = h

  if any_changed:
    with open(os.path.join(output_dir, "hit_type"+str(hit_type_id)+".json"), 'w') as f:
      json.dump(hit_type, f)
  if new_hit_available:
    return hit_add_assignment(new_hit_available, hit_type, hit_type_id, curr, output_dir, worker_id, submit_url)
  elif not in_progress_assignment is None:
    in_progress_assignment['accept_time'] = curr
    with open(os.path.join(output_dir, "hit"+str(in_progress_hit['mturk_id'])+".json"), 'w') as f:
      json.dump(in_progress_hit, f)
    return in_progress_assignment
  elif not almost_finished_hit is None:
    return hit_add_assignment(almost_finished_hit, hit_type, hit_type_id, curr, output_dir, worker_id, submit_url)
  else:
    return None
''')

    with open(os.path.join(self.output_dir, 'cgi-bin', 'annotate'+str(hit_type_id)+'.py'), 'w') as f: 
      f.write('''#!/usr/bin/env python
from new_assignment import *
import os
import Cookie
import datetime
import json
import time
import numpy as np
import filelock

hit_type_id = "'''+str(hit_type_id)+'''"
output_dir = "'''+self.output_dir+'''"
url = "'''+urlparse.urlsplit(self.url).netloc+'''"
submit_url = "'''+self.url + '/cgi-bin/submit'+str(hit_type_id)+'.py' +'''"
lock = filelock.FileLock(os.path.join(output_dir, "lock"+str(hit_type_id)+".json"))
with lock.acquire():
  try:
    cookie = Cookie.SimpleCookie(os.environ["HTTP_COOKIE"])
    worker_id = cookie["worker_id"].value
  except (Cookie.CookieError, KeyError):
    # Add new worker
    with open(os.path.join(output_dir, "hit_type"+hit_type_id+".json"), 'r') as f:
      hit_type = json.load(f)
    worker_id = len(hit_type['workers'])
    hit_type['workers'][worker_id] = os.environ["REMOTE_ADDR"]
    with open(os.path.join(output_dir, "hit_type"+hit_type_id+".json"), 'w') as f:
      json.dump(hit_type, f)
    
    expiration = datetime.datetime.now() + datetime.timedelta(days=30)
    cookie = Cookie.SimpleCookie()
    cookie["worker_id"] = worker_id
    cookie["worker_id"]["domain"] = url
    cookie["worker_id"]["path"] = "/"
    cookie["worker_id"]["expires"] = expiration.strftime("%a, %d-%b-%Y %H:%M:%S PST")
    
  a = new_assignment(output_dir, worker_id, hit_type_id)
  print "Content-type: text/html"
  print cookie.output()
  print
  if a:
    args = ""
    if not '?' in a['external_url']:
      args = '?hitTypeId='+str(hit_type_id)+'&hitId='+str(a['hit_id'])+'&workerId='+str(worker_id)+'&turkSubmitTo='+submit_url
    print '<html><frameset cols="100%"><frame src="'+a['external_url']+args+'"></frame></frameset></html>'
  else:
    print '<html>No more HITs left</html>'
        ''')
    os.chmod(os.path.join(self.output_dir,  'cgi-bin', 'annotate'+str(hit_type_id)+'.py'), 0775)


    with open(os.path.join(self.output_dir,  'cgi-bin', 'submit'+str(hit_type_id)+'.py'), 'w') as f: 
      f.write('''#!/usr/bin/env python
from new_assignment import *
import os
import Cookie
import datetime
import json
import cgi
import time
import filelock

output_dir = "'''+self.output_dir+'''"
params = cgi.FieldStorage()
annotate_url = "'''+self.url + '/cgi-bin/annotate'+str(hit_type_id)+'.py' +'''"
submit_url = "'''+self.url + '/cgi-bin/submit'+str(hit_type_id)+'.py' +'''"
hit_type_id = "'''+str(hit_type_id)+'''"

success = False
lock = filelock.FileLock(os.path.join(output_dir, "lock"+str(hit_type_id)+".json"))
with lock.acquire():
  if 'hitId' in params:
    with open(os.path.join(output_dir, "hit"+str(params['hitId'].value)+".json"), 'r') as f:
      hit = json.load(f)
    for a in hit['assignments']:
      if a['status'] == 'In Progress' or a['status'] == 'Expired' and a['worker_id']==params['workerId'].value:
        answers = {}
        for k in params.keys(): answers[k] = params.getvalue(k)
        a['answers'] = answers
        a['status'] = 'Submitted'
        a['submit_time'] = time.time()
        with open(os.path.join(output_dir, "hit"+str(params['hitId'].value)+".json"), 'w') as f:
          json.dump(hit, f)
        success = True
        break
  
  if success:
    a = new_assignment(output_dir, params['workerId'].value, hit_type_id)
    print "Content-type: text/html\"
    print
    if a: 
      args = ""
      if not '?' in a['external_url']:
        args = '?hitTypeId='+str(hit_type_id)+'&hitId='+str(a['hit_id'])+'&workerId='+str(worker_id)+'&turkSubmitTo='+submit_url
      print '<html>Loading next HIT...<script language="javascript">window.location.href = "'+a['external_url']+args+'"</script></html>'
    else:
      print '<html>No more HITs left</html>'
    print
    #print "HTTP/1.1 302 Found"
    #print "Location: "+annotate_url+"\\r\\n"
    #print "Connection: close\\r\\n\\n"
  else:
    print "Status: 400\\n\\n"
  
''')
    os.chmod(os.path.join(self.output_dir,  'cgi-bin', 'submit'+str(hit_type_id)+'.py'), 0775)
    
    return rs

  def create_hit(self, hit_type=None, question=None, lifetime=3600*24*7, max_assignments=1):
    with open(os.path.join(self.output_dir, "hit_type"+str(hit_type)+".json"), 'r') as f:
      ht = json.load(f)
    rs = ResultSet()
    hit_id = len(ht['hits'])
    #if question:
    #  matchObj = re.match( r'.*/hits/(.*).html', question.external_url, re.M|re.I)
    #  if matchObj and matchObj.groups() and len(matchObj.groups()) > 0:
    #    hit_id = matchObj.group(1)
    rs.append(HitResponse(HITId=hit_id))
    hit = {'mturk_id':rs[0].HITId, 'hit_type':hit_type, 'external_url':question.external_url, 'assignments':[], 'lifetime':lifetime, 'status':'Assignable', 'max_assignments':max_assignments,'create_time':time.time()}
    ht['hits'].append([rs[0].HITId, max_assignments, 0, [], 0])
    with open(os.path.join(self.output_dir, "hit_type"+str(hit_type)+".json"), 'w') as f:
      json.dump(ht, f)
    os.chmod(os.path.join(self.output_dir, "hit_type"+str(hit_type)+".json"), 0666)
    with open(os.path.join(self.output_dir, "hit"+str(hit['mturk_id'])+".json"), 'w') as f:
      json.dump(hit, f)
    os.chmod(os.path.join(self.output_dir, "hit"+str(hit['mturk_id'])+".json"), 0666)
    return rs
  
  def get_assignments(self, hit_id=None, page_size=None):
    rs = ResultSet()
    with open(os.path.join(self.output_dir, "hit"+str(hit_id)+".json"), 'r') as f:
      hit = json.load(f)
    for a in hit['assignments']:
      if a['status'] == 'Submitted':
        answers = []
        for k in a['answers']:
          ans = QuestionFormAnswer(None)
          ans.fields = [a['answers'][k]]
          ans.qid = k
          answers.append(ans)
        #answers = [type("QuestionFormAnswer",(object,),dict(fields=[a['answers'][k]],qid=k)) for k in a['answers']]
        ass = Assignment(AssignmentId=str(a['hit_id'])+'_'+str(a['worker']), WorkerId=a['worker'], HITId=a['hit_id'], AssignmentStatus=a['status'], Deadline=a['accept_time']+hit['lifetime'], AcceptTime=a['accept_time'], SubmitTime=a['submit_time'], answers=[answers])
        rs.append(ass)
    return rs
    
  def disable_hit(self, hit_id=None):
    with open(os.path.join(self.output_dir, "hit"+str(hit_id)+".json"), 'r') as f:
      hit = json.load(f)
    hit['status'] = 'Disposed'
    with open(os.path.join(self.output_dir, "hit"+str(hit['mturk_id'])+".json"), 'w') as f:
      json.dump(hit, f)
    return ResultSet()
  
  def dispose_hit(self, hit_id=None):
    with open(os.path.join(self.output_dir, "hit"+str(hit_id)+".json"), 'r') as f:
      hit = json.load(f)
    hit['status'] = 'Disposed'
    with open(os.path.join(self.output_dir, "hit"+str(hit['mturk_id'])+".json"), 'w') as f:
      json.dump(hit, f)
    return ResultSet()
  
  def expire_hit(self, hit_id=None):
    with open(os.path.join(self.output_dir, "hit"+str(hit_id)+".json"), 'r') as f:
      hit = json.load(f)
    hit['lifetime'] = 0
    with open(os.path.join(self.output_dir, "hit"+str(hit['mturk_id'])+".json"), 'w') as f:
      json.dump(hit, f)
    return ResultSet()

  def extend_hit(self, hit_id=None, assignments_increment=None, expiration_increment=None):
    with open(os.path.join(self.output_dir, "hit"+str(hit_id)+".json"), 'r') as f:
      hit = json.load(f)
    if expiration_increment: hit['lifetime'] += expiration_increment
    if assignments_increment: hit['max_assignments'] += assignments_increment
    with open(os.path.join(self.output_dir, "hit"+str(hit['mturk_id'])+".json"), 'w') as f:
      json.dump(hit, f)
    return ResultSet()

  def set_reviewing(self, revert=False):
    with open(os.path.join(self.output_dir, "hit"+str(hit_id)+".json"), 'r') as f:
      hit = json.load(f)
    hit['status'] = 'Reviewable' if revert else 'Reviewing'
    with open(os.path.join(self.output_dir, "hit"+str(hit['mturk_id'])+".json"), 'w') as f:
      json.dump(hit, f)
    return ResultSet()
  
  def get_balance(self):
    rs = ResultSet()
    rs.append(0)
    return rs

def StressTestLocalCrowdsourcer(base_url, dataset):
  rand_perms = {}
  for i in dataset.images:
    rand_perms[i] = np.random.permutation(len(dataset.images[i].z))
  lock = threading.Lock()
  for w in dataset.workers:
    worker_thread = threading.Thread(target=StressTestWorker, args=(base_url, dataset, w, rand_perms, lock))
    worker_thread.start()

def StressTestWorker(base_url, dataset, workerId, rand_perms, lock):
  sleepTime = 1
  while True:
    response = urllib2.urlopen('http://'+base_url+'/cgi-bin/annotate0.py')
    html = response.read()
    if 'Annotation task finished' in html:
      print 'Worker thread ' + str(workerId) + ' exiting'
      break
    elif 'No more HITs left' in html:
      print 'Worker thread ' + str(workerId) + ' waiting for batch to complete'
    else:
      matchObj = re.match( r'.*src="(.*)".*', html, re.M|re.I) 
      if not matchObj or not matchObj.groups() or len(matchObj.groups()) == 0:
        print 'Worker thread ' + str(workerId) + ' error parsing annotate.py'
      else:
        hitUrl = matchObj.group(1)
        matchObj = re.match( r'(.*)_(.*)\..*?hitTypeId=(.*)&hitId=(.*)&workerId=(.*)&turkSubmitTo=(.*)', hitUrl, re.M|re.I)
        if not matchObj or not matchObj.groups() or len(matchObj.groups()) == 0:
          print 'Worker thread ' + str(workerId) + ' error parsing ' + hitUrl
        else:
          hitId = matchObj.group(4)
          response = urllib2.urlopen(hitUrl)
          html = response.read()
          matchObj = re.search( r'.*;\n\ngParameters = (.*);\nif\(typeof.*\n', html, re.M|re.I)
          if not matchObj or not matchObj.groups() or len(matchObj.groups()) == 0:
            print 'Worker thread ' + str(workerId) + ' error parsing ' + hitUrl + ' contents'
          else:
            params = json.loads(matchObj.group(1))
            data = {'dataset':dataset.encode(), 'images':{}, 'annos':[]}
            data['dataset'] = dataset.encode()
            for i in params['image_ids']:
              #data['images'][i] = dataset.images[i].encode()
              with lock:
                w = dataset.images[i].workers[rand_perms[i][0]]
                rand_perms[i] = rand_perms[i][1:]
              data['annos'].append({'image_id':i, 'worker_id':w, 'anno':dataset.images[i].z[w].encode()})
            
            post_args = {'assignmentId':0 , 'workerId':w, 'hitId':hitId, 'hitTypeId':0, 'json':json.dumps(data)}
            print 'Worker thread ' + str(workerId) + ' POST for ' + hitUrl + ': ' + json.dumps(post_args)
            r = requests.post('http://'+base_url+'/cgi-bin/submit0.py', data=post_args)
            print 'Worker thread ' + str(workerId) + ' POST ' + hitUrl + ' response is ' + str(r.status_code) + ' ' + str(r.reason)
            sleepTime = 1
            continue
    
    time.sleep(sleepTime)
    sleepTime = min(60, 2*sleepTime)

    
    
  
  
