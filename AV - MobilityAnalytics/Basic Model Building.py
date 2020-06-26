# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:34:36 2020

@author: jayasans4085
"""

import pandas as pd
import numpy as np

import os
os.chdir(r'C:\Users\jayasans4085\OneDrive - ARCADIS\Documents\Learning\Other Hackathons\Analytics-Vidhya-Competitions\AV - MobilityAnalytics\Data')

dt = pd.read_csv('Cleaned Data.csv')
dt.head()
dt.drop(columns=['Unnamed: 0'],inplace = True)
dt.set_index(['Trip_ID'],inplace = True)

#%% Train Test Split
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(dt[[i for i in list(dt) if i!='Surge_Pricing_Type']], dt['Surge_Pricing_Type'], test_size = 0.2, random_state = 0)

#%%
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(xTrain, yTrain)

predt_y = classifier.predict(xTrain)
pred1_y = classifier.predict(xTest)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(yTrain,predt_y))
print(classification_report(yTest,pred1_y))

print(confusion_matrix(yTrain,predt_y))
print(confusion_matrix(yTest,pred1_y))
