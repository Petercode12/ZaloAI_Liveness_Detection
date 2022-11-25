import pandas as pd
import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import wget
import zipfile
from skimage import feature
from utils import LocalBinaryPatterns, N, size
import threading

if not os.path.exists('train'):
  print("DOWNLOADING DATA:")
  response = wget.download('https://dl-challenge.zalo.ai/liveness-detection/train.zip', "train.zip")

  print("\nEXTRACTING DATA:")
  trainzip = zipfile.ZipFile('train.zip')
  trainzip.extractall()
else:
  print("DATA HAS BEEN DOWNLOADED")
      
data_list = pd.read_csv('./train/label.csv')
data_list = [ (x,y) for (x,y) in zip(data_list['fname'],data_list['liveness_score'])]



print("\nPREPROCESSING DATA:")
X_=[]
y_=[]
for f, label in tqdm(data_list):
  vid_capture = cv2.VideoCapture(f'./train/videos/{f}')
  X = []
  y = label
  count = 0
  while(vid_capture.isOpened()):
    ret, frame = vid_capture.read()
    if count%(vid_capture.get(7)//(N+1))==0:
      frame = cv2.resize(frame, (size[1], size[0]))
      X.append(frame.tolist())
    if len(X) == N:
      X = np.array(X).astype('uint8')
      X_.append(X)
      y_.append(y)
      break
    count += 1
X_ = np.array(X_).astype('uint8')
y_ = np.array(y_)
with open(f'./train/data_pickle_{N}_{size[0]}_{size[1]}_3.pkl', 'wb') as file:
  pickle.dump((X_, y_), file)

class LocalBinaryPatterns:
  def __init__(self, numPoints, radius):
    # store the number of points and radius
    self.numPoints = numPoints
    self.radius = radius
  def describe(self, image, eps=1e-7):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(image, self.numPoints,
      self.radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
      bins=np.arange(0, self.numPoints + 3),
      range=(0, self.numPoints + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    # return the histogram of Local Binary Patterns
    return hist

def map_function(x, numPoints, radius):
  desc = LocalBinaryPatterns(numPoints, radius)
  try:
    idx = x.numpy()
  except:
    idx = x
  X, y = X_[idx], y_[idx]
  X = [desc.describe(i) for i in X.astype('uint8')]
  # X = np.mean(np.array(X), axis=0)
  X = np.array(X).reshape((-1))
  return X, np.array([y])

X=[]
y=[]
print("\nPREPARING DATA FOR TRAINING:")
for idx in tqdm(range(len(X_))):
  X_item, y_item = map_function(idx, 98, 2)
  X.append(X_item)
  y.append(y_item)
X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)
with open(f'./train/data_pickle_{N}_{size[0]}_{size[1]}_3_phase_2.pkl', 'wb') as file:
  pickle.dump((X, y), file)