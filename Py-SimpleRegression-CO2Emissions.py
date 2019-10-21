#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab as py
#For Matplotlib graphs to be inline and included in jupyter notebook, next to the code.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Download data to Jupyter work folder
get_ipython().system('wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv')


# In[12]:


# Read and describe data
df = pd.read_csv("FuelConsumption.csv")
df.head()
#df.describe()


# In[11]:


# Explore - histogram
viz = df[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()


# In[13]:


# Scatter plot to know Co2 vs fuel Consumption
plt.scatter(df.FUELCONSUMPTION_COMB, df.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()


# In[15]:


# Scatter plot to know Co2 vs Engine size
plt.scatter(df.ENGINESIZE, df.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[16]:


# Scatter plot to know CO2 vs cylinders
plt.scatter(df.CYLINDERS, df.CO2EMISSIONS, color = 'blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()


# In[17]:


# Build training and test datasets 70% training, 30% test
msk = np.random.rand(len(df)) < 0.7
train = df[msk]
test = df[~msk]


# In[18]:


# Build model
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[20]:


# Plot the modelled line
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[22]:


# Evaluate accuracy of model
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Residual sum of squares (MSE): %.3f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.3f" % r2_score(test_y_hat , test_y) )


# In[ ]:




