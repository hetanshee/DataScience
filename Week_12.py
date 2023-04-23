#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np

class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, num_iterations=200):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for i in range(self.num_iterations):
            # Forward propagation
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # Backward propagation
            dw = (1/m) * np.dot(X.T, (y_predicted - y))
            db = (1/m) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return np.round(y_predicted)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# In[22]:


import pandas as pd


df = pd.read_csv("creditcard.csv")


# In[23]:


X = df.loc[:, df.columns != 'Class']
y = df["Class"]


# In[24]:


import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # Custom Model

# In[25]:



custom = LogisticRegressionCustom()

custom.fit(X_train,y_train)

y_pred_custom = custom.predict(X_test)


# In[26]:


from sklearn.metrics import accuracy_score

custom_score = accuracy_score(y_test, y_pred_custom)

custom_score


# # Inbuilt Model

# In[27]:



from sklearn.linear_model import LogisticRegression

inbuilt = LogisticRegression(random_state=42,solver='lbfgs', max_iter=200).fit(X_train, y_train)

y_pred_inbuilt = inbuilt.predict(X_test)


# In[28]:


from sklearn.metrics import accuracy_score

inbuilt_score = accuracy_score(y_test, y_pred_inbuilt)

inbuilt_score


# In[29]:


inbuilt_score - custom_score


# ## There is a difference of 8.30e-4 between the custom model and scikit's inbuilt model

# In[ ]:




