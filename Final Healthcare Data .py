#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pycaret


# In[2]:


import pandas as pd


# In[3]:


dataset=pd.read_csv("C:\\Users\\vitis\\OneDrive\\Desktop\\4 TH SEM MBA\\LIVE PROJECT\\cad.csv")#scaled for max vlaues in excel itself

# check the shape of data
dataset.shape


# In[4]:


# sample 5% of data to be used as unseen data
data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
# print the revised shape
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# In[5]:


pip install scikit-learn==0.23.2


# In[6]:


from pycaret.classification import *
s = setup(data = dataset, target = 'Type', session_id=123)


# In[7]:


best_model = compare_models()


# In[8]:


print(best_model)


# In[9]:


# check available models
models()


# In[10]:


dt = create_model('dt')


# In[11]:


# trained model object is stored in the variable 'dt'. 
print(dt)


# In[12]:


tuned_dt = tune_model(dt)


# In[13]:


# tuned model object is stored in the variable 'tuned_dt'. 
print(tuned_dt)


# In[14]:


plot_model(tuned_dt, plot = 'auc')


# In[15]:


plot_model(tuned_dt, plot = 'pr')


# In[16]:


plot_model(tuned_dt, plot='feature')


# In[17]:


plot_model(tuned_dt, plot = 'confusion_matrix')


# In[18]:


evaluate_model(tuned_dt)


# In[19]:


predict_model(tuned_dt);


# In[20]:


#Finalize Model for Deployment
# finalize rf model
final_dt = finalize_model(tuned_dt)
# print final model parameters
print(final_dt)


# In[21]:


predict_model(final_dt);


# In[22]:


unseen_predictions = predict_model(final_dt, data=data_unseen)
unseen_predictions.head()


# In[32]:


# check metric on unseen data
from pycaret.utils import check_metric
check_metric(unseen_predictions['Type'], unseen_predictions['Label'], metric = 'Accuracy')


# In[26]:


# saving the final model
save_model(final_dt,'Final Control Healthcare')


# In[27]:


# loading the saved model
saved_final_dt = load_model('Control Healthcare')


# In[28]:


# predict on new data
new_prediction = predict_model(saved_final_dt, data=data_unseen)
new_prediction.head()


# In[29]:


#Model interpretability
import shap
from itertools import chain
from collections import defaultdict
pd.set_option('display.max_columns', None)
shap.initjs()


# In[30]:


interpret_model(final_dt)


# In[31]:


interpret_model(final_dt,plot = 'correlation')


# In[ ]:


pip install shap


# In[ ]:


sudo apt-get install python3.7


# In[ ]:




