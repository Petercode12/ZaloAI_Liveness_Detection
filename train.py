from skimage import feature
import numpy as np
import cv2
import pickle
import tqdm
from sklearn.model_selection import train_test_split 
from sklearn.metrics import make_scorer, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calculate_eer(y_true, y_score):
    '''
    Returns the equal error rate for a binary classifier output.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

with open(f'./train/data_pickle_1_280_160_3_phase_2.pkl', 'rb') as file:
  X, y = pickle.load(file)

data_train, data_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=4)

model.fit(data_train, labels_train)

from sklearn.metrics import accuracy_score
labels_predict = model.predict(data_train)
print(f"\nTraining accuracy: {accuracy_score(labels_train, labels_predict)}")
print(f"Training EER: {calculate_eer(labels_train, labels_predict)}")


from sklearn.metrics import accuracy_score
labels_predict = model.predict(data_test)
print(f"\nTesting accuracy: {accuracy_score(labels_test, labels_predict)}")
print(f"Testing EER: {calculate_eer(labels_test, labels_predict)}")

pickle.dump(model, open('./RandomForestClassifier.pkl', 'wb'))