#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


import pickle
import pandas as pd


# In[3]:


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[4]:


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[5]:


df = read_data("fhv_tripdata_2021-04.parquet")


# In[6]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


# In[9]:


print("Mean is {0}".format(y_pred.mean()))


# In[10]:


# year = 2021
# month = 2


# # In[11]:


# df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# # In[13]:


# df.sample(3)


# # In[16]:


# df_result = df[["ride_id", "duration"]]


# # In[18]:


# output_file = f'result_{year:04d}-{month:02d}.parquet'


# # In[19]:


# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )


# # In[ ]:




