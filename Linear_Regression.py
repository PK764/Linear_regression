#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Before implementing the linear regression, we need to download dataset into our local repository


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(20.0,10.0)

#Reading data
data=pd.read_csv("C:/Users/konda/Downloads/Data_Science_DataSets/headbrain.csv") #reading file from local repository
print(data.shape)
data.head() 


# In[5]:


#collecting x and y
X= data['Head Size(cm^3)'].values
Y= data['Brain Weight(grams)'].values


# # Calculating the cofficients of the line

# In[8]:


# Mean X and Y
#line equation=> y=mx+c m=sigmaof((x-meanOfx)(y-meanOfy))/sigmaOf((x-meanOfx)^2)
mean_x= np.mean(X)
mean_y= np.mean(Y)
#print(mean_x,mean_y)
n=len(X)
#print(n)
numer = 0
denom = 0
for i in range(n):
    numer +=(X[i]-mean_x) * (Y[i] - mean_y) 
    denom +=(X[i]-mean_x) **2
m=numer/denom  
c=mean_y - (m*mean_x)

print(m,c)

    


# # plotting the values

# In[21]:


#plotting values and regression line
max_x = np.max(X) +100
min_x = np.min(X) -100

#calculating line values x and y 
x= np.linspace(min_x,max_x,1000)
y= c+m*x

#PLOTTING LINE
plt.plot(x,y, color='#58b970',label='regression Line')

#plotting scatter points
plt.scatter(X,Y, c='#ef5423', label='Scatter plot')
plt.legend()
plt.show()


# # Calculating mean Square error

# In[24]:


ss_t = 0
ss_r = 0
for i in range(n):
    y_predicted= c+m*X[i]
    ss_t +=(Y[i]-mean_y)**2
    ss_r +=(Y[i]-y_predicted)**2
r2=1-(ss_r/ss_t)
print(r2)


# # easy way of calculating mean square error using sklearn

# In[30]:


#using scikitlearn_ calculating r^2 value
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
X=X.reshape((m,1))
#creating model
reg= LinearRegression()
#fitting training data   
reg=reg.fit(X,Y)

#y-prediction
Y_pred=reg.predict(X)
r2=reg.score(X,Y)
print(r2)


# In[ ]:




