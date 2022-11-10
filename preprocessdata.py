import pandas as pd
import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import wget
import zipfile

print("DOWNLOADING DATA:")
response = wget.download('https://dl-challenge.zalo.ai/liveness-detection/train.zip', "train.zip")

print("\nEXTRACTING DATA:")
trainzip = zipfile.ZipFile('train.zip')
trainzip.extractall()

data_list = pd.read_csv('./train/label.csv')
data_list = [ (x,y) for (x,y) in zip(data_list['fname'],data_list['liveness_score'])]

N = 10
size = (280,160, 3)
vector_size = 256

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
with open(f'.train/data_pickle_10_280_160_3.pkl', 'wb') as file:
  pickle.dump((X_, y_), file)