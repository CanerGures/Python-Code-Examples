import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

team=pd.read_csv('team.csv',encoding='latin-1')
x_team = team.iloc[:,4].values
y_team = team.iloc[:,6].values
z_team = team.iloc[:,7].values
S_team = team.iloc[:,8].values
x_team = x_team.reshape(-1,1)
y_team = y_team.reshape(-1,1)
z_team = z_team.reshape(-1,1)
S_team = S_team.reshape(-1,1)

matx=np.concatenate([x_team.T,y_team.T,z_team.T])
X=matx.T

X=np.append(arr=np.ones((18,1)).astype(int),values=X,axis=1)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X,S_team)
s_pred=reg.predict(X)
u=S_team-s_pred


if reg.coef_[0][1] > reg.coef_[0][2]:
    if reg.coef_[0][1] > reg.coef_[0][3]:
        print("1. Age")
        if reg.coef_[0][2] > reg.coef_[0][3]:
            print("2. Exp")
            print("3. Power")
        else:
            print("2. Power")
            print("3. Exp")

if reg.coef_[0][2] > reg.coef_[0][1]:
    if reg.coef_[0][2] > reg.coef_[0][3]:
        print("1. Exp")
        if reg.coef_[0][1] > reg.coef_[0][3]:
            print("2. Age")
            print("3. Power")
        else:
            print("2. Power")
            print("3. Age")

if reg.coef_[0][3] > reg.coef_[0][1]:
    if reg.coef_[0][3] > reg.coef_[0][2]:
        print("1. Power")
        if reg.coef_[0][1] > reg.coef_[0][2]:
            print("2. Age")
            print("3. Exp")
        else:
            print("2. Exp")
            print("3. Age")

plt.scatter(s_pred,u,color="black",s=10,label="Train Data")
plt.plot(reg.predict(X),X,color="orange")
plt.legend(loc='upper right')
plt.title("Residual Error Plot")
plt.show()
