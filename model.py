import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split,RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\monul\OneDrive\Desktop\30 April 2022\Datasets\drinks.csv")
df.pop("country")
df.pop("continent")

x = df.drop("total_litres_of_pure_alcohol",axis = 1)
y = df["total_litres_of_pure_alcohol"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

scalar = StandardScaler()
scalar.fit(x_train)
arr = scalar.transform(x_train)

Rf_reg = RandomForestRegressor(random_state=2)
Rf_reg.fit(x_train, y_train)

Rf_reg = RandomForestRegressor(random_state=2)
hyp = {'n_estimators':np.arange(10,150),'criterion':['mse','mae'],'max_depth':np.arange(5,15),'min_samples_split':np.arange(5,20),'min_samples_leaf':np.arange(4,15),'max_features':['sqrt','log2']}
Rscv_Rf_reg = RandomizedSearchCV(Rf_reg, hyp, cv=5)
Rscv_Rf_reg.fit(x_train, y_train)
Rscv_Rf_reg.best_params_

Rf_reg = RandomForestRegressor(criterion='mse', max_depth=13, max_features='log2', min_samples_leaf=5, min_samples_split=8, n_estimators=106,random_state=2)
Rf_reg.fit(x_train,y_train)                               

with open("Alcohol.pkl","wb") as file:
    pickle.dump(Rf_reg,file)
    
with open ("Al_obj.obj","wb") as file1:
    pickle.dump(scalar,file1)