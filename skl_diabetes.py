#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# sex -> patient's gender (1: male; 0: female)
# 
# HighCol -> 0 = no high cholesterol 1 = high cholesterol
# 
# CholCheck -> 0 = no cholesterol check in 5 years 1 = yes cholesterol check in 5 years
# 
# HeartDiseaseorAttack -> coronary heart disease (CHD) or myocardial infarction (MI) 0 = no 1 = yes
# 
# PhysActivity -> physical activity in past 30 days - not including job 0 = no 1 = yes
# 
# Fruits -> Consume Fruit 1 or more times per day 0 = no 1 = yes
# 
# Veggies -> Consume Vegetables 1 or more times per day 0 = no 1 = yes
# 
# HvyAlcoholConsump -> (adult men >=14 drinks per week and adult women >=7 drinks per week) 0 = no 1 = yes
# 
# GenHlth -> Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 
# 5 = poor
# 
# MentHlth -> days of poor mental health scale 1-30 days
# 
# PhysHlth -> physical illness or injury days in past 30 days scale 1-30
# 
# DiffWalk -> Do you have serious difficulty walking or climbing stairs? 0 = no 1 = yes
# 
# Stroke -> you ever had a stroke. 0 = no, 1 = yes
# 
# HighBP -> 0 = no high, BP 1 = high BP
# 
# Diabetes -> 0 = no diabetes, 1 = diabetes

# In[2]:


df = pd.read_csv('./diabetes_data.csv')
df


# In[3]:


X = df.drop(columns='Diabetes').to_numpy()
y = df['Diabetes'].to_numpy()

X_train, X_, y_train, y_ = train_test_split(X,y, test_size=0.40, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_,y_, test_size=0.50, random_state=42)


# In[4]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_cv = scaler.transform(X_cv)
X_test = scaler.transform(X_test)


# In[5]:


model = LogisticRegression(C=0.009, penalty='elasticnet', solver='saga', max_iter=10000, l1_ratio=0.5)
model.fit(X_train, y_train)


# In[6]:


y_pred = model.predict(X_cv)
print(model.score(X_cv,y_cv))


# In[7]:


y_pred = model.predict(X_test)
print(model.score(X_test,y_test))


# In[8]:


import joblib

joblib.dump({
    model : "model",
    scaler : "scaler"        
    }, "Diabetes.pkl")

