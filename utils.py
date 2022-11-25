import cv2
import pickle
from sklearn.svm import *
import numpy as np
from skimage import feature

N = 10
size = (280,160)

try:
  svm = pickle.load(open('./RandomForestClassifier.pkl', 'rb'))  
except:
  pass

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

def file2class(filename):
  vid_capture = cv2.VideoCapture(filename)
  X = []
  count = 0
  while(vid_capture.isOpened()):
    ret, frame = vid_capture.read()
    if count%(vid_capture.get(7)//(N+1))==0:
      frame = cv2.resize(frame, (size[1], size[0]))
      X.append(frame.tolist())
    if len(X) == N:
      X = np.array(X).astype('uint8')
      break
  desc = LocalBinaryPatterns(98, 2)
  X = [desc.describe(i) for i in X.astype('uint8')]
  X = np.array(X).reshape((1,-1))
  # return svm.predict(X)[0]
  return svm.predict_proba(X)[0][1]