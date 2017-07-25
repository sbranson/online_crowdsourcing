import json
import yaml
import urllib
import os
import time
from crowdsourcing.annotation_types.part import *

MAX_DOWNLOAD_RETRIES = 10

# Import binary classification dataset from Peter Welinder's CUBAM dataset (bluebirds/ducks)
def import_dataset_cubam(worker_yaml_file_in, json_file_out, gt_yaml_file_in=None, base_url="http://sbranson.no-ip.org/bluebirds", image_dir=None):
  with open(worker_yaml_file_in,"r") as f:
    data = yaml.load(f)
  if not gt_yaml_file_in is None:
    with open(gt_yaml_file_in,"r") as f:
      data['gt'] = yaml.load(f)
    
  if image_dir and not os.path.exists(image_dir):
    os.makedirs(image_dir)
  
  enc = {'images':{}, 'annos':[], 'gt_labels':[], 'dataset':{'gtype':'binary'}}
  for w in data:
    for i in data[w]:
      if not i in enc['images']:
        enc['images'][str(i)] = {'id':str(i), 'url':base_url+'/'+str(i)+'.jpg'}
        if not image_dir is None: 
          enc['images'][str(i)]['fname'] = image_dir + "/" + str(i) + ".jpg"
          if not os.path.isfile(enc['images'][str(i)]['fname']):
            download_image(enc['images'][str(i)]['url'], enc['images'][str(i)]['fname'])
        
      if w=='gt':
        enc['gt_labels'].append({'image_id':str(i), 'label':{'label':data[w][i]}})
      else:
        enc['annos'].append({'image_id':str(i), 'worker_id':str(w), 'anno':{'label':data[w][i]}})
  
  with open(json_file_out, 'w') as f:
      json.dump(enc, f)
  

# Import binary classification dataset from Grant's data, which is a dump of CTurk 
# and expert annotations on a 40 class subset of CUB
def import_dataset_vibe(json_file_in, dir_out, image_dir=None):
  with open(json_file_in,"r") as f:
    data = json.load(f)
  if image_dir and not os.path.exists(image_dir):
    os.makedirs(image_dir)
  
  classes = []
  for class_name in data:
    enc = {'images':{}, 'annos':[], 'gt_labels':[], 'dataset':{'gtype':'binary'}}
    for d in data[class_name]['worker_labels']:
      i, w, a, gt = str(d['image_id']), str(d["worker_id"]), d["answer"], d["gt_answer"]
      
      if not i in enc['images']:
        enc['images'][i] = {'id':i, 'url':d["vibe_url"]}
        enc['gt_labels'].append({'image_id':i, 'anno':{'label':gt}})
        if not image_dir is None: 
          enc['images'][i]['fname'] = image_dir + "/" + str(i) + ".jpg"
          if not os.path.isfile(enc['images'][i]['fname']):
            download_image(enc['images'][i]['url'], enc['images'][i]['fname'])
      enc['annos'].append({'image_id':i, 'worker_id':w, 'anno':{'label':a}})
      
    with open(dir_out + '/class' + str(len(classes)) + '.json', 'w') as f:
      json.dump(enc, f)
    classes.append(class_name)
  
  with open(dir_out + '/classes.json', 'w') as f:
    json.dump(classes, f)


def import_bbox_dataset_old_server(worker_json_file_in, json_file_out, expert_json_file_in=None, image_dir=None):
  with open(worker_json_file_in,"r") as f:
    data = json.load(f)
  if not expert_json_file_in is None:
    with open(expert_json_file_in,"r") as f:
      ex_data = json.load(f)
      for a in ex_data['annotations']:
        a['worker'] = 'gt'
        data['annotations'].append(a)

  if image_dir and not os.path.exists(image_dir):
    os.makedirs(image_dir)
  
  images, annos, gt_labels = {}, [], []
  for i in data['images']:
    images[i] = {"id":i, "url":data['images'][i]['url'], "original_name":data['images'][i]['original_name'], 
                 "width":data['images'][i]["width"], "height":data['images'][i]["height"]}
    if not image_dir is None:
      images[i]['fname'] = image_dir + "/" + str(i) + ".jpg"
      if not os.path.isfile(images[i]['fname']):
        download_image(images[i]['url'], images[i]['fname'])

  for a in data['annotations']:
    for i in a['answer']['images']:
        w, l, width, height = a['worker'], a['answer']['images'][i], images[i]['width'], images[i]['height']
        boxes = []  
        for bb in l:
          if bb is None: continue
          b = bb[1]
          box = {'name':bb[0], 'x':min(width-1,max(0,float(b['x'])-float(b['scaleX'])/2)),
                 'y':min(height-1,max(0,float(b['y'])-float(b['scaleY'])/2)),
                 'x2':min(width-1,max(0,float(b['x'])+float(b['scaleX'])/2)),
                 'y2':min(height-1,max(0,float(b['y'])+float(b['scaleY'])/2)),
                 'timestamp':b['timestamp'],'image_width':width, 'image_height':height
          }
          boxes.append(box)
   
        if w=='gt': gt_labels.append({'image_id':i, 'image_width':width, 'image_height':height, 'label':{'bboxes':boxes}})
        else: annos.append({'image_id':i, 'worker_id':w, 'image_width':width, 'image_height':height, 'anno':{'bboxes':boxes}})
        

  enc = {'images':images, 'annos':annos, 'dataset':{'gtype':'bboxes'}}
  if len(gt_labels)>0: 
    enc["gt_labels"] = gt_labels
  with open(json_file_out, 'w') as f:
    json.dump(enc, f)

def import_part_dataset_old_server(worker_json_file_in, json_file_out, expert_json_file_in=None, image_dir=None, part_names=None):
  with open(worker_json_file_in,"r") as f:
    data = json.load(f)
  if not expert_json_file_in is None:
    with open(expert_json_file_in,"r") as f:
      ex_data = json.load(f)
      for a in ex_data['annotations']:
        a['worker'] = 'gt'
        data['annotations'].append(a)

  if image_dir and not os.path.exists(image_dir):
    os.makedirs(image_dir)
  
  images, annos, gt_labels = {}, [], []
  for i in data['images']:
    images[i] = {"id":i, "url":data['images'][i]['url'], "original_name":data['images'][i]['original_name']}
    if not image_dir is None:
      images[i]['fname'] = image_dir + "/" + str(i) + ".jpg"
      if not os.path.isfile(images[i]['fname']):
        download_image(images[i]['url'], images[i]['fname'])

  min_view = 100000
  for a in data['annotations']:
    for i in a['answer']['images']:
      if len(a['answer']['images'][i])>0:
        for b in a['answer']['images'][i][0]:
          if 'view' in b:
            min_view = min(min_view, int(b['view']))
  
  num_parts = 0
  if not part_names: part_names = {}
  for a in data['annotations']:
    for i in a['answer']['images']:
      if len(a['answer']['images'][i])>0:
        w, l = a['worker'], a['answer']['images'][i][0]

        bbox = [float("inf"), float("inf"), float("-inf"), float("-inf"), 0]
        num_vis = 0
        part_locs = {}
        for j in range(0,len(l)):
          p = -1
          if 'part' in l[j]:
            if not l[j]['part'] in part_names:
              part_names[l[j]['part']] = len(part_names)
            p = part_names[l[j]['part']]  
            name = l[j]['part']
          elif 'view' in l[j]:
            p = (int(l[j]['view'])-min_view)/3
            name = part_names[p] if len(part_names) else p
          if p < 0:
            continue
          num_parts = max(num_parts, p+1)
          part_locs[p] = {'x':float(l[j]['x']), 'y':float(l[j]['y']), 'vis':bool(l[j]['visible']), 'color':PART_COLORS[p%len(PART_COLORS)], 'outline_color':PART_OUTLINE_COLORS[p%len(PART_OUTLINE_COLORS)]}
          if name: part_locs[p]['name'] = name
          if part_locs[p]['vis']:
            bbox[0] = float(min(bbox[0], part_locs[p]['x']))
            bbox[1] = float(min(bbox[1], part_locs[p]['y']))
            bbox[2] = float(max(bbox[2], part_locs[p]['x']))
            bbox[3] = float(max(bbox[3], part_locs[p]['y']))
            num_vis += 1
        
        part_locs_a = [{} for p in range(len(part_names))]
        for p in part_locs: part_locs_a[p] = part_locs[p]
        if num_vis == 0:
          bbox = [0, 0, 1e-7, 1e-7, 0]
        else:
          bbox[0] = max(0, bbox[0])
          bbox[1] = max(0, bbox[1])
          bbox[2] = max(1e-7, bbox[2]-bbox[0])
          bbox[3] = max(1e-7, bbox[3]-bbox[0])
          bbox[4] = num_vis

        if w=='gt': gt_labels.append({'image_id':i, 'label':{'parts':part_locs_a, 'bbox':bbox}})
        else: annos.append({'image_id':i, 'worker_id':w, 'anno':{'parts':part_locs_a, 'worker_bbox':bbox}})
  
  enc = {'images':images, 'annos':annos, 'dataset':{'part_names':part_names,'gtype':'keypoints'}}
  if len(gt_labels)>0: 
    enc["gt_labels"] = gt_labels
  
  with open(json_file_out, 'w') as f:
    json.dump(enc, f)
  

  
def download_image(url, fname, max_download_retries=10):
  error = True
  if not os.path.isfile(fname):
    print "Downloading " + url + " to " + fname + "..."
    error = False
    for i in range(max_download_retries):
      try:
        urllib.urlretrieve(url, fname)
        with open(fname) as f:
          s = f.read()
          if "<Code>InternalError</Code>" in s: 
            print s
            error = True
      except Exception,e:
        print e
        error = True
      if error:
        time.sleep(1)
      else: 
        break
  return error
