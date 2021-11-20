#!/usr/bin/env python
# coding: utf-8

# # Case Study 2
# 
# #### Predicting Central Neuropathic Pain (CNP) in people with Spinal Cord Injury (SCI) from Electroencephalogram (EEG) data.
# 
# * CNP is pain in response to non-painful stimuli, episodic (electric shock), “pins and needles”, numbness
# * There is currently no treatment, only prevention
# * Preventative medications have strong side-effects
# * Predicting whether a patient is likely to develop pain is useful for selective treatment
# 
# #### Task
# Your task is to devise a feature engineering strategy which, in combination with a classifier of your choice, optizimes prediction accuracy.
# 
# #### Data
# The data is preprocessed brain EEG data from SCI patients recorded while resting with eyes closed (EC) and eyes opened (EO).
# * 48 electrodes recording electrical activity of the brain at 250 Hz 
# * 2 classes: subject will / will not develop neuropathic pain within 6 months
# * 18 subjects: 10 developed pain and 8 didn’t develop pain
# * the data has already undergone some preprocessing
#   * Signal denoising and normalization
#   * Temporal segmentation
#   * Frequency band power estimation
#   * Normalization with respect to total band power
#   * Features include normalized alpha, beta, theta band power while eyes closed, eyes opened, and taking the ratio of eo/ec.
# * the data is provided in a single table ('data.csv') consisting of 
#   * 180 rows (18 subjects x 10 repetitions), each containing
#   * 432 columns (9 features x 48 electrodes)
#   * rows are in subject major order, i.e. rows 0-9 are all samples from subject 0, rows 10-19 all samples from subject 1, etc.
#   * columns are in feature_type major order, i.e. columns 0-47 are alpha band power, eyes closed, electrodes 0-48
#   * feature identifiers for all columns are stored in 'feature_names.csv'
#   * 'labels.csv' defines the corresponding class (0 or 1) to each row in data.csv
# 
# #### Objective Measure
# Leave one subject out cross-validation accuracy, sensitivity and specificity.
# 
# #### Report
# Report on your feature engineering pipeline, the classifier used to evaluate performance, and the performance as mean and standard deviation of accuracy, sensitivity and specificity across folds. Give evidence for why your strategy is better than others.
# 
# 
# 

# In[26]:


import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[18]:


# load data
# rows in X are subject major order, i.e. rows 0-9 are all samples from subject 0, rows 10-19 all samples from subject 1, etc.
# columns in X are in feature_type major order, i.e. columns 0-47 are alpha band power, eyes closed, electrodes 0-48
# feature identifiers for all columns in X are stored in feature_names.csv
X = np.loadtxt('data.csv', delimiter=',') 
y = np.loadtxt('labels.csv', delimiter=',')
with open('feature_names.csv') as f:
    csvreader = csv.reader(f, delimiter=',')
    feature_names = [row for row in csvreader][0]
data_df = pd.DataFrame(X)
data_df.columns = feature_names
target_array = np.array(y)
print(target_array)

#reference: https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html
#目标数组
#除了特征矩阵 X 之外，我们通常还使用标签或目标数组.
#按照惯例，我们通常将其称为 y.
#目标数组通常是一维的，长度为 n_samples，通常包含在 NumPy 数组或 Pandas 系列中.
#目标数组可能有连续的数值，或离散的类/标签.
#虽然一些 Scikit-Learn 估计器确实以二维 [n_samples, n_targets] 目标数组的形式处理多个目标值.
#但我们将主要处理一维目标数组的常见情况。


# In[17]:


# plotting data in 2D with axes sampled 
# a) at random 
# b) from same electrode
# c) from same feature type
num_features = 9
num_electrodes = 48

# a) indices drawn at random
i0, i1 = np.random.randint(0, X.shape[1], size=2)

# b) same electrode, different feature (uncomment lines below)
f0, f1 = np.random.randint(0, num_features, size=2)
e = np.random.randint(0, num_electrodes)
i0_electrode, i1_electrode = f0*num_electrodes + e, f1*num_electrodes + e

# b) same feature, different electrode (uncomment lines below)
f = np.random.randint(0, num_features)
e0, e1 = np.random.randint(0, num_electrodes, size=2)
i0_feature, i1_feature = f*num_electrodes + e0, f*num_electrodes + e1


def plotting(i0,i1):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    colors = ['blue', 'red']
    # select features i0, i1 and separate by class
    X00, X01 = X[y==0][:,i0], X[y==1][:,i0]
    X10, X11 = X[y==0][:,i1], X[y==1][:,i1]
    # plot cumulative distribution of feature i0 separate for each class
    axes[0].hist(X00, bins=20, label='y=0, '+ feature_names[i0], density=True, alpha=0.5)
    axes[0].hist(X01, bins=20, label='y=1, '+ feature_names[i0], density=True, alpha=0.5)
    axes[0].hist(X10, bins=20, label='y=0, '+ feature_names[i1], density=True, alpha=0.5)
    axes[0].hist(X11, bins=20, label='y=1, '+ feature_names[i1], density=True, alpha=0.5)
    axes[0].set_title('histograms')
    axes[0].legend()
    axes[1].plot(np.sort(X00), np.linspace(0,1,X00.shape[0]), label='y=0, '+ feature_names[i0], alpha=0.5)
    axes[1].plot(np.sort(X01), np.linspace(0,1,X01.shape[0]), label='y=1, '+ feature_names[i0], alpha=0.5)
    axes[1].plot(np.sort(X10), np.linspace(0,1,X10.shape[0]), label='y=0, '+ feature_names[i1], alpha=0.5)
    axes[1].plot(np.sort(X11), np.linspace(0,1,X11.shape[0]), label='y=1, '+ feature_names[i1], alpha=0.5)
    axes[1].set_title('empirical cumulative distribution functions')
    axes[1].legend()
    axes[2].scatter(X00, X10, label='y=0')
    axes[2].scatter(X01, X11, label='y=1')
    axes[2].set_xlabel(feature_names[i0])
    axes[2].set_ylabel(feature_names[i1])
    axes[2].set_title('scatter plot')
    axes[2].legend()
    
plotting(i0,i1)
plotting(i0_electrode,i1_electrode)
plotting(i0_feature,i1_feature)
#相同电极差异最小


# In[30]:


#方差选择法
#原始数据shape
print(data_df.shape)
# Feature Selection with variance
data_with_variance = VarianceThreshold(threshold=3).fit_transform(data_df)
#使用方差选择法检验后的shape
#选了两个最优的，这个参数可以调
print(data_with_variance.shape)


# In[29]:


#卡方检验
#原始数据shape
print(data_df.shape)
X_new = SelectKBest(chi2, k=2).fit_transform(data_df, target_array)
#使用过滤法的卡方检验后的shape
#选了两个最优的，这个参数可以调
print(X_new.shape)


# In[ ]:


#相关系数法
#code....


# In[ ]:


#互信息法
#code....


# In[ ]:


#RFE
#不可用，因为数据集问题


# In[ ]:


#基于惩罚项的特征选择法
#code


# In[ ]:


#基于树模型的特征选择法
#不知道可不可以做成树模型


# In[ ]:


#需要画图或者其它方式来验证这些方法的效率还准确度
#code


