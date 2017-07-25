import flickrapi
import json
import math
import os
import urllib

def FlickrImageSearch(queryTerms, out_dir, api_key, api_secret_key, max_photos=4000, min_taken_date=None, max_taken_date=None, license=None, privacy_filter=1, content_type=1, media='photos', image_size='c'):
  api = flickrapi.FlickrAPI(api_key, api_secret_key)

  # Do a Flickr API image search for each query term
  max_results = 0
  qResults = {}
  page_size = 100
  for queryTerm in queryTerms:
    query = queryTerm
    results = []
    for page in range(int(math.ceil(max_photos/float(page_size)))):
      res = json.loads(api.photos_search(text=query, min_taken_date=min_taken_date, max_taken_date=max_taken_date, license=license, privacy_filter=privacy_filter, content_type=content_type, media=media, page=page+1, format='json'))
      if 'photos' in res and 'photo' in res['photos']:
        for p in res['photos']['photo']: 
          p['query'] = query
          p['url'] = 'http://farm1.staticflickr.com/{0}/{1}_{2}_{3}.jpg'.format(p['server'], p['id'], p['secret'], image_size)
        results += res['photos']['photo']
      else:
        break
    max_results = max(max_results, len(results)) 
    qResults[queryTerm] = results
  
  # Merge results, removing duplicate responses
  uniqueResults = {}
  uniqueResultsA = []
  for i in range(max_results):
    for queryTerm in queryTerms:
      if i < len(qResults[queryTerm]) and not qResults[queryTerm][i]['id'] in uniqueResults:
        uniqueResults[qResults[queryTerm][i]['id']] = qResults[queryTerm][i]
        uniqueResultsA.append(qResults[queryTerm][i])
  
  if not os.path.isdir(os.path.join(out_dir, 'images')): 
    os.makedirs(os.path.join(out_dir, 'images'))
  with open(os.path.join(out_dir, 'flickr.json'), 'w') as f:
    json.dump(uniqueResultsA, f)
  
  # Download images
  for ri in range(min(max_photos, len(uniqueResultsA))):
    r = uniqueResultsA[ri]
    url = 'http://farm1.staticflickr.com/{0}/{1}_{2}_{3}.jpg'.format(r['server'], r['id'], r['secret'], image_size)
    urllib.urlretrieve(url, os.path.join(out_dir, 'images', str(ri) + '.jpg'))
  
