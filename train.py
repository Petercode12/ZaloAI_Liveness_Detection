from skimage import feature
import numpy as np
import cv2
import pickle
import tqdm
from sklearn.model_selection import train_test_split 

with open(f'./train/data_pickle_10_280_160_3.pkl', "rb") as input_file:
  X_,y_ = pickle.load(input_file)
  
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

data_train, data_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import *
model = NuSVC(random_state=42, kernel = 'rbf')
model.fit(data_train, labels_train)

from sklearn.metrics import accuracy_score
labels_predict = model.predict(data_train)
print(f"\nTraining accuracy: {accuracy_score(labels_train, labels_predict)}")

from sklearn.metrics import accuracy_score
labels_predict = model.predict(data_test)
print(f"Testing accuracy: {accuracy_score(labels_test, labels_predict)}")

pickle.dump(model, open('./SVM_10_2.pkl', 'wb'))