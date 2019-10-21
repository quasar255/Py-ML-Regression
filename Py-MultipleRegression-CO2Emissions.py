#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Download data to folder
get_ipython().system('wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv')


# In[3]:


# Explore data
df = pd.read_csv("FuelConsumption.csv")
df.head()


# In[5]:


# Explore correlation
print("Correlation b/w Engine size and CO2 is: ", np.corrcoef(df.ENGINESIZE, df.CO2EMISSIONS))
plt.scatter(df.ENGINESIZE, df.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[7]:


# create training(80%) and test(20%) datasets
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]


# In[8]:


# Run multiple regression
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)


# In[10]:


y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])

# Evaluate MSE
print("Residual sum of squares: %.2f" % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


# In[16]:


# Calculate value for Eng size 2, 4 cylinders, fuel consumption 8.5
print(regr.predict([[2,4,8.5]]))


# In[ ]:




