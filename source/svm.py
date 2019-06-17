import pandas as pd
import numpy as np

import glob, os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn import svm
from sklearn.model_selection import cross_val_score

# == Read data into df
dirname=os.path.dirname
THIS_FOLDER = dirname(os.path.abspath(__file__))
ROOT_FOLDER = dirname(THIS_FOLDER)
table = pd.DataFrame()

file_path = glob.glob("data/UCI_Credit_Card.csv")[0]

print("Reading csv file...")
df = pd.read_csv( os.path.join(ROOT_FOLDER, file_path) )

# == Preprocessing:
print("Preprocessing....")

X = df.drop(columns=["ID", "default.payment.next.month"])
y = df["default.payment.next.month"]

# features that need One-Hot Encoding
columns_need_onehot_encoding = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
# features that need normalizing
columns_need_normalizing = [i for i in X.columns.values if i not in columns_need_onehot_encoding]
high_correlation_features = ["PAY_0"]

# columns that need one-hot encoding
X_1 = X[ columns_need_onehot_encoding ]
X_1 = X_1[ [i for i in columns_need_onehot_encoding if i in high_correlation_features] ]
X_1 = X_1.values

# columns that need normalizing
X_2 = X[ columns_need_normalizing ]
X_2 = X_2[ [i for i in columns_need_normalizing if i in high_correlation_features] ]
X_2 = X_2.values

# label encode (make data positive and continuous)
X_1 = pd.DataFrame(X_1).apply(LabelEncoder().fit_transform).values

# one-hot encode
X_1 = OneHotEncoder().fit_transform(X_1).toarray()

# normalize 
if X_2.shape[1]>0:
    X_2 = MinMaxScaler().fit_transform(X_2)

# concate two array
X = np.concatenate((X_1, X_2), axis=1)

# == Validating
print("Cross validating...")

clf = svm.LinearSVC(random_state=0)
scores = cross_val_score(clf,X_1,y,cv=10,scoring='accuracy',error_score=np.nan)
print("# SVM (linear) mean accuracy (10-fold): %f" % scores.mean())

clf = svm.SVC(kernel="poly", gamma='scale', C=1000, degree=3)
scores = cross_val_score(clf,X_1,y,cv=10,scoring='accuracy',error_score=np.nan)
print("# SVM (polynomial) mean accuracy (10-fold): %f" % scores.mean())