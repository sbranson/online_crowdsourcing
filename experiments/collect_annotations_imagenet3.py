import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import json
from mpl_toolkits.mplot3d import Axes3D

#sys.path.append(os.path.join(os.path.dirname(__file__),"../"))
from crowdsourcing.interfaces.mechanical_turk import *
from crowdsourcing.interfaces.local_webserver import *
from crowdsourcing.util.image_search import *
from crowdsourcing.annotation_types.classification import *
from crowdsourcing.annotation_types.bbox import *
from crowdsourcing.annotation_types.part import *

# directory containing the images we want to annotate
IMAGE_DIR = 'data/classification/imagenet'

OUTPUT_FOLDER = 'ImageNet4'

USE_MTURK = True
ONLINE = True
WORKERS_PER_IMAGE = 0

with open('keys.json') as f: keys = json.load(f)

# Amazon account information for paying for mturk tasks, see 
# http://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSGettingStartedGuide/AWSCredentials.htm
AWS_ACCESS_KEY = keys.AWS_ACCESS_KEY
AWS_SECRET_ACCESS_KEY = AWS_SECRET_ACCESS_KEY
SANDBOX = False

# API key for Flickr image search, see https://www.flickr.com/services/api/misc.api_keys.html
FLICKR_API_KEY = keys.FLICKR_API_KEY
FLICKR_API_SECRET_KEY = keys.FLICKR_API_SECRET_KEY
MAX_PHOTOS = 4000

HOST = 'sbranson.no-ip.org'


# The name of the objects we want to collect.  Images will be obtained by crawling flickr
# image search for each object, and we want to use mturk to filter out images that
# don't contain the object of interest
CLASSES = [ { 'object_name' : 'beaker', 'definition' : 'A flatbottomed jar made of glass or plastic; used for chemistry', 'search' : ['beaker', 'beaker chemistry', 'beaker lab'], 'wikipedia_url' : 'https://en.wikipedia.org/wiki/Beaker_(glassware)', 'example_image_urls' : ['http://imagenet.stanford.edu/nodes/12/02815834/99/998d93ef3fdd9a30034cda8f0ce246b7bb13ebc4.thumb', 'http://imagenet.stanford.edu/nodes/12/02815834/51/5171dde0d020b00923d4297d88d427326846efb2.thumb', 'http://imagenet.stanford.edu/nodes/12/02815834/d0/d06ccaf38a410e0b59bfe73819eb7bd0028bb8f1.thumb', 'https://sbranson.no-ip.org/online_crowdsourcing/not_beaker.jpg' ] },
            { 'object_name' : 'scorpion', 'definition' : 'Arachnid of warm dry regions having a long segmented tail ending in a venomous stinger', 'search' : ['scorpion', 'scorpion arachnid'], 'wikipedia_url' : 'https://en.wikipedia.org/wiki/Scorpion', 'example_image_urls' : ['http://imagenet.stanford.edu/nodes/2/01770393/b0/b02dcf2c1d8c7a735b52ab74300c342124e4be5c.thumb', 'http://imagenet.stanford.edu/nodes/2/01770393/31/31af6ea97dd040ec2ddd6ae86fe1f601ecfc8c02.thumb', 'http://imagenet.stanford.edu/nodes/2/01770393/38/382e998365d5667fc333a7c8f5f6e74e3c1fe164.thumb', 'http://imagenet.stanford.edu/nodes/2/01770393/88/88bc0f14c9779fad2bc364f5f4d8269d452e26c2.thumb'] },
            { 'object_name' : 'apiary', 'definition' : 'A shed containing a number of beehives', 'search' : ['apiary'], 'wikipedia_url' : 'https://en.wikipedia.org/wiki/Apiary', 'example_image_urls' : ['http://imagenet.stanford.edu/nodes/10/02727426/1f/1f6f71add82d10edad8b3630ec26490055c70a5d.thumb', 'http://imagenet.stanford.edu/nodes/10/02727426/94/94a3624ff3e639fe2d8ae836e91ca7e8fcdd0ed7.thumb', 'http://imagenet.stanford.edu/nodes/10/02727426/15/15a37da46bddd5010d3f1d1996899b8472c9556b.thumb', 'http://imagenet.stanford.edu/nodes/10/02727426/01/013b499a063b6ea83218c5ed63ea811bce5a9974.thumb'] },
            { 'object_name' : 'cardigan', 'definition' : 'Knitted jacket that is fastened up the front with buttons or a zipper', 'search' : ['cardigan'], 'wikipedia_url' : 'https://en.wikipedia.org/wiki/Cardigan_(sweater)', 'example_image_urls' : ['http://imagenet.stanford.edu/nodes/9/02963159/d7/d7419041a96e8baf9a870c81d549ad0b345c8127.thumb', 'http://imagenet.stanford.edu/nodes/9/02963159/34/34256aaf7b10073ec16dc5ddb0b31305878de875.thumb', 'http://imagenet.stanford.edu/nodes/9/02963159/e8/e8a50045dd40da5299ee8817052edfc090b05355.thumb', 'http://imagenet.stanford.edu/nodes/9/02963159/38/38216bf40fafe4bb526fabb430188c24b968a152.thumb'] } 
          ]

for c in CLASSES:
  # directories to store images  and results
  output_folder = os.path.join(OUTPUT_FOLDER, c['object_name'])
  image_folder = os.path.join('output', output_folder, 'flickr')
  if not os.path.exists(image_folder):
    os.makedirs(image_folder)
  
  # Download images from Flickr
  FlickrImageSearch(c['search'], image_folder, FLICKR_API_KEY, FLICKR_API_SECRET_KEY, max_photos=MAX_PHOTOS)

  # Load an unlabelled dataset by scanning a directory of images
  dataset = CrowdDatasetBinaryClassification(name=c['object_name'])
  dataset.scan_image_directory(os.path.join(image_folder, 'images'))

  if USE_MTURK:
    crowdsource = MTurkCrowdsourcer(dataset, AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY, HOST, output_folder, sandbox=SANDBOX,
                                    hit_params = c, online = ONLINE, thumbnail_size = (100,100), initial_assignments_per_image=WORKERS_PER_IMAGE)
  else:
    crowdsource = LocalCrowdsourcer(dataset, HOST, output_folder, hit_params = c, online = ONLINE, thumbnail_size = (100,100), initial_assignments_per_image=WORKERS_PER_IMAGE, port=8080)
  crowdsource.run()

