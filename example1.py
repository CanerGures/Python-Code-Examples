import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.svm import SVC

data = pd.read_csv('Grand-slams-men-2013.csv')

X = data.iloc[:, [12,30]]
y = data.iloc[:, [3]].sum(axis=1)

X_train = X.iloc[0:200]
X_test = X.iloc[200:, ]
y_train = y.iloc[0:200]
y_test = y.iloc[200:, ]

label_lst = []
score_lst = []

def RunSVM(kernel):
 svr = SVC(kernel=kernel)

 label = 'SVR(rbf)'
 mdl = SVR(kernel='rbf', gamma='scale')
 print(label)
 label_lst.append(label)

 label = 'SVR(linear)'
 mdl = SVR(kernel='linear', gamma='scale')
 print(label)
 label_lst.append(label)

 mdl = SVR(kernel='poly', gamma='scale')
 print(label)
 label_lst.append(label)

 svr.fit(X_train, y_train)
 predicted = svr.predict(X_test)

 plt.figure()
 for i in range(len(predicted)):
        if predicted[i] == 0:
            plt1 = plt.scatter(X_test.values[i, 0], X_test.values[i, 1], c='red')

        else:
            plt2 = plt.scatter(X_test.values[i, 0], X_test.values[i, 1], c='blue')

 plt.title(kernel)


RunSVM('linear')
RunSVM('poly')
RunSVM('rbf')

plt.show()