
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import time as t
from scipy import stats

import os
import sys

start_initial = t.time()


# In[2]:


x_train_path = sys.argv[1]
train_path = os.path.abspath(x_train_path)
path_train = os.path.dirname(train_path)
os.chdir(path_train)

x_test_path = sys.argv[2]
test_path = os.path.abspath(x_test_path)
path_test = os.path.dirname(test_path)
os.chdir(path_test)

x_train = (pd.read_csv(x_train_path, header = None, na_filter = False, low_memory = False)).values
x_test = (pd.read_csv(x_test_path, header = None, na_filter = False, low_memory = False)).values

y_train = x_train[:,0].reshape((x_train.shape[0],1))
x_train = x_train[:,1:]
x_test = x_test[:,1:]

label = np.full((x_train.shape[0], 1), -1)

col_mean = x_train.mean(axis=0)
x_train = x_train[:,(col_mean != 0)]
x_test = x_test[:,(col_mean != 0)]

col_mean = x_train.mean(axis=0)
x_train = x_train - col_mean
col_std = x_train.std(axis=0)
for i in range(x_train.shape[1]):
    if(col_std[i] != 0):
        x_train[:,i] = x_train[:,i]/(col_std[i])

col_mean = x_test.mean(axis=0)
x_test = x_test - col_mean
col_std = x_test.std(axis=0)
for i in range(x_test.shape[1]):
    if(col_std[i] != 0):
        x_test[:,i] = x_test[:,i]/(col_std[i])


# In[3]:


x_train_zero = x_train - np.mean(x_train, axis = 0)
M = np.cov(x_train_zero.T)

eig_n = int(sys.argv[4])

lamda, v = np.linalg.eig(M)

idx = lamda.argsort()[::-1]

lamda = lamda[idx]
v = v[:,idx]

lamda = lamda[:eig_n]
v = v[:,:eig_n]

x_train_new = (v.T).dot(x_train.T).T
x_test_new = (v.T).dot(x_test.T).T


# In[4]:


# Number of clusters
k = 10


# In[5]:


# start = t.time()

# cluster = []

# random.seed(1574)
# a = random.randint(0,x_train.shape[0])
# cluster.append(x_train[a,:])

# distances = np.zeros((x_train.shape[0],k))

# for j in range(k-1):
#     check = []
#     for i in range(x_train.shape[0]):
#         distances[i,j] = np.linalg.norm(cluster[j]-x_train[i])
#         check.append(np.min(distances[i,:j+1]))
#     cluster.append(x_train[np.argmax(check)])
#     print(str(j) + " clusters initialized")


# print("Time take to make kmeans++ initialized cluster - " + str(t.time()-start))


# In[6]:


start = t.time()

cluster = []

for i in range(0,x_train_new.shape[0],round(x_train_new.shape[0]/k)):
    cluster.append(np.mean(x_train_new[i:i+round(x_train_new.shape[0]/k),:], axis = 0))
    
print("Time take to make random initialized cluster - " + str(t.time()-start))


# In[7]:


cluster = np.array(cluster)


# In[8]:


print(cluster.shape)
print(label.shape)


# In[9]:


# WITHOUT DICTIONARY

# start = t.time()
# steps = 0
# old_cluster = cluster*100

# while(np.max(np.linalg.norm(old_cluster-cluster)) > k/50):
#     old_cluster = np.copy(cluster)
#     for i in range(x_train.shape[0]):
#         distances = []
#         for j in range(k):
#             distances.append(np.linalg.norm(x_train[i,:]-cluster[j]))
#         label[i] = np.argmin(distances)
# #         print("Example number - " + str(i) + ", Class number - " + str(np.argmax(distances)))
#     for i in range(k):
#         if(np.sum(label[:,0] == i) > 0):
#             cluster[i] = np.mean(x_train[label[:,0] == i], axis = 0)
#     print("Max shift in any centroid is " + str(np.max(np.linalg.norm(old_cluster-cluster))))
# #     for i in range(k):
# #         print("Class " + str(i) + " has " + str(np.sum(label[:,0] == i)) + " examples")
# #     print("---------------------------------------------------------------------------") 
#     steps += 1

# print("Number of iterations - " + str(steps))
# print("Time taken to train - " + str(t.time()-start))


# In[10]:


# WITH DICTIONARY

start = t.time()
steps = 0
old_cluster = cluster*100

while(np.max(np.linalg.norm(old_cluster-cluster)) > k/15):
    d_dict = {}
    for i in range(k):
        d_dict[str(i)] = []
    
    old_cluster = np.copy(cluster)
    
    for i in range(x_train_new.shape[0]):
        distances = []
        for j in range(k):
            distances.append(np.linalg.norm(x_train_new[i,:]-cluster[j]))
        a = np.argmin(distances)
        label[i] = a
        d_dict[str(a)].append(x_train_new[i])
#         print("Example number - " + str(i) + ", Class number - " + str(np.argmax(distances)))
    
    for i in range(k):
        cluster[i] = np.mean(d_dict[str(i)], axis = 0)
    
    print("Max shift in any centroid is " + str(np.max(np.linalg.norm(old_cluster-cluster))))
#     for i in range(k):
#         print("Class " + str(i) + " has " + str(np.sum(label[:,0] == i)) + " examples")
#     print("---------------------------------------------------------------------------") 
    steps += 1

print("Number of iterations - " + str(steps))
print("Time taken to train - " + str(t.time()-start))


# In[11]:


# from pylab import *
# for i in range(cluster.shape[0]):
#     imshow(cluster[i,:].reshape((28,28)))
#     figure(i)


# In[12]:


class_label = np.full((k,1), -1)

for i in range(k):
    class_label[i] = stats.mode(y_train[label==i])[0]


# In[13]:


predictions = np.full((x_test_new.shape[0],1), 0)

for i in range(x_test_new.shape[0]):
    distances = []
    for j in range(k):
        distances.append(np.linalg.norm(x_test_new[i,:]-cluster[j]))
#     print("Distance array - " + str(distances))
#     print("Minimum distance - " + str(np.argmin(distances)))
#     print(class_label[np.argmin(distances)])
    predictions[i] = class_label[np.argmin(distances)]


# In[15]:


print("Total time taken to run - " + str(t.time() - start_initial))

x_output = sys.argv[3]
actual_path_out = os.path.abspath(x_output)
path_out = os.path.dirname(actual_path_out)
os.chdir(path_out)

np.savetxt(x_output, predictions)

