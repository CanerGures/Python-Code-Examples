from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("dataset.csv")

X = data.iloc[:, [6, 10,11,12,13,14,16]]
y = data.iloc[:, [19,20,21,22,23]].sum(axis=1)

X_Train = X.iloc[0:200]
X_Test = X.iloc[200:,]
y_Train = y.iloc[0:200]
y_Test = y.iloc[200:,]

def RunRandForest():
    arr_mse1 = []
    arr_mse2 = []
    arr_mse3 = []


    for i in range(200):

        reg = RandomForestRegressor(max_depth=1, n_estimators=i+1, max_features=4)
        reg = reg.fit(X_Train, y_Train.ravel())
        yPredicted = reg.predict(X_Test)
        mse = np.mean((y_Test - yPredicted) * (y_Test - yPredicted))
        arr_mse1.append(mse)

        reg = RandomForestRegressor(max_depth=1, n_estimators=i+1, max_features="auto")
        reg = reg.fit(X_Train, y_Train.ravel())
        yPredicted = reg.predict(X_Test)
        mse = np.mean((y_Test - yPredicted) * (y_Test - yPredicted))
        arr_mse2.append(mse)

        reg = RandomForestRegressor(max_depth=1, n_estimators=i+1, max_features="sqrt")
        reg = reg.fit(X_Train, y_Train.ravel())
        yPredicted = reg.predict(X_Test)
        mse = np.mean((y_Test - yPredicted) * (y_Test - yPredicted))
        arr_mse3.append(mse)

    reg = RandomForestRegressor(max_depth=7, n_estimators=200, max_features=4)
    reg = reg.fit(X_Train, y_Train.ravel())
    yPredicted1 = reg.predict(X_Test)
    err1 = y_Test - yPredicted

    reg = RandomForestRegressor(max_depth=1, n_estimators=200, max_features=4)
    reg = reg.fit(X_Train, y_Train.ravel())
    yPredicted2 = reg.predict(X_Test)
    err2 = y_Test - yPredicted

    horizontal = []
    for i in range(0,200):
        horizontal.append(i+1)

    plt.figure()
    plt.plot(horizontal, arr_mse1, color="blue")
    plt.plot(horizontal, arr_mse2, color="red")
    plt.plot(horizontal, arr_mse3, color="green")

    plt.xlabel("0-200 scalar")
    plt.ylabel("MSE")
    plt.figure()

    plt.scatter(yPredicted1, err1, color="red")

    plt.scatter(yPredicted2, err2, color ="blue")
    plt.xlabel("Estimation")
    plt.ylabel("Error of Estimation")
   # plt.hlines(y=0, xmin=0, xmax=40000, linewidth=2)
    plt.show()


RunRandForest()