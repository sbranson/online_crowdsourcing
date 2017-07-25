import pickle
import os
import numpy as np
from binary import * 
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

class BinaryComputerVisionPredictor(object):
  def __init__(self, feature_extractor, num_splits=4, computer_vision_cache=None):
    self.feature_extractor = feature_extractor
    self.num_splits = num_splits
    self.computer_vision_cache = computer_vision_cache
  
  def predict_probs(self, images, labels, valid_train=None, cache_name=None, cv_worker=None, naive=False):
    # Compute image features
    image_names = [im.fname for im in images]
    if self.computer_vision_cache: cache_name = self.computer_vision_cache
    if cache_name is None or not os.path.isfile(cache_name):
      print "Extracting features for " + str(len(image_names)) + " images to " + cache_name  + "..."
      features = self.feature_extractor.extract_features([os.path.abspath(f) for f in image_names])
      '''
      features_d = {}
      for i in range(len(image_names)): features_d[image_names[i]] = features[i]
      with open(cache_name, 'wb') as f: 
        pickle.dump(features_d, f)
      '''
      np.savez(cache_name+'.npz', features=features)
      with open(cache_name, 'wb') as f:
        pickle.dump(image_names, f)
    else:
      features = np.load(cache_name+'.npz')['features']
      with open(cache_name, 'rb') as f:
        image_names_cached = pickle.load(f)
        image_names_d = {image_names_cached[i]:i for i in range(len(image_names_cached))}
        features_d = {image_names_cached[i]:features[i,:] for i in range(len(image_names_cached))}
      image_names_new = []
      for i in image_names:
        if not i in image_names_d:
          image_names_new.append(i)
      if len(image_names_new):  # Handle the case where new images have been added since features were cached
        features = self.feature_extractor.extract_features(image_names_new)
        for i in range(len(image_names_new)): features_d[image_names_new[i]] = features[i]
        with open(cache_name, 'wb') as f: 
          pickle.dump(image_names, f)
        features = np.asarray([features_d[image_names[i]] for i in range(len(image_names))])
        np.savez(cache_name+'.npz', features=features)
      else:
        features = np.asarray([features_d[image_names[i]] for i in range(len(image_names))])
    
    perm_inds = np.random.permutation(len(image_names))
    probs = {}
    v = np.asarray(valid_train) if valid_train else np.asarray([True for i in range(len(image_names))])
    for n in range(self.num_splits):
      start_ind = (n*len(image_names))/self.num_splits 
      end_ind = ((n+1)*len(image_names))/self.num_splits 
      sp, ep, test_inds = perm_inds[:start_ind], perm_inds[end_ind:], perm_inds[start_ind:end_ind]
      train_inds = np.concatenate((sp[v[sp]], ep[v[ep]]))
      val_inds = test_inds[v[test_inds]]
      
      Y_train = np.asarray([labels[i].label for i in train_inds])
      p = [(.5,.5) for i in range(len(test_inds))]
      if len(train_inds)>0 and (Y_train==0).sum()>0 and (Y_train==1).sum()>0 and len(val_inds)>0:
        Y_val = np.asarray([labels[i].label for i in val_inds])
        if (Y_val==0).sum()>0 and (Y_val==1).sum()>0:
          X_val = np.asarray([features[i] for i in val_inds])
          X_train = np.asarray([features[i] for i in train_inds])
          X_test = np.asarray([features[i] for i in test_inds])
          Y_test = np.asarray([labels[i].label for i in test_inds])
          print "Train: " + str(len(train_inds)) + "," + str((Y_train==0).sum()) + " neg," + str((Y_train==1).sum()) + " pos"
          clf = svm.LinearSVC()
          clf.fit(X_train, Y_train)
          clf_prob = CalibratedClassifierCV(clf, cv="prefit", method='sigmoid')
          clf_prob.fit(X_val, Y_val)
          p = clf_prob.predict_proba(X_test)
          print "Val: " + str(len(val_inds)) + "," + str((Y_val==0).sum()) + " neg," + str((Y_val==1).sum()) + " pos, min_p=" + str(np.asarray([p[i][1] for i in range(len(val_inds))]).min()) + " max_p=" + str(np.asarray([p[i][1] for i in range(len(val_inds))]).max())
      for i in range(len(test_inds)):
        probs[image_names[test_inds[i]]] = p[i][1]
    
    retval = []
    for i in range(len(images)):
      cv_pred = CrowdLabelBinaryClassification(images[i], cv_worker, label=(1.0 if probs[image_names[i]]>.5 else 0.0))
      cv_pred.prob = probs[image_names[i]]
      retval.append(cv_pred)
    return retval
