import pandas as pd
import numpy as np

import glob, os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans
import sklearn.metrics as sm

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
high_correlation_features = ["LIMIT_BAL", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

# columns that need one-hot encoding
X_1 = X[ [i for i in columns_need_onehot_encoding if i in high_correlation_features] ]
X_1 = X_1.values

# columns that need normalizing
X_2 = X[ [i for i in columns_need_normalizing if i in high_correlation_features] ]
X_2 = X_2.values

# label encode (make data positive and continuous)
X_1 = pd.DataFrame(X_1).apply(LabelEncoder().fit_transform).values

# one-hot encode
X_1 = OneHotEncoder().fit_transform(X_1).toarray()

# normalize 
if X_2.shape[1]>0:
    X_2 = MinMaxScaler().fit_transform(X_2)

# concate X_1 and X_2
X = np.concatenate((X_1, X_2), axis=1)


# == Validating:
print("Validating...")

def kmeans_validate(n):
    model = KMeans(n_clusters=n)
    model.fit(X)

    # Y[column 0]:group, Y[column 1]:target 
    Y = np.array((model.labels_, y.values)).T
    group_labels = []
    for i in range(n):
        tmp_arr = Y[ Y[:,0]==i ]
        default_count = [0,0]
        default_count[0] = len( tmp_arr[ tmp_arr[:,1]==0 ] )
        default_count[1] = len( tmp_arr[ tmp_arr[:,1]==1 ] )
        if default_count[0]>default_count[1]:
            group_labels.append(0)
        else:
            group_labels.append(1)

    predict_y = []
    for v in model.labels_:
        predict_y.append( group_labels[v] )

    print("# Kmeans' with %d group accuracy: %f" % (n,sm.accuracy_score(y, predict_y)) )

for n in [2, 4, 8, 16, 32]:
    kmeans_validate(n)