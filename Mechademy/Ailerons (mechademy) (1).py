#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# # Loading the Dataset

# In[94]:


df=pd.read_csv("ailerons_(mechademy).csv")
df.head()


# top 5 heads of the datasets.

# In[4]:


df.tail()


# Bottom 5 of the dataset.

# # Analyzing the Patterns of the Dataset

# In[6]:


# Checking shape
df.shape


# 12250 rows and 41 columns are present in the dataset.

# In[7]:


# Checking Datatypes
df.info()


# 37 float datatype columns and 4 integer datatype columns are present in the dataset.

# In[8]:


# Checking null values
df.isnull().sum()


# Null values are not present in the dataset.

# In[13]:


# Checking columns
df.columns


# In[24]:


# checking unique values in some of the columns that seems to be nominal or single value 
print(pd.unique(df[['SeTime1', 'SeTime2',
       'SeTime3', 'SeTime4', 'SeTime5', 'SeTime6', 'SeTime7', 'SeTime8',
       'SeTime9', 'SeTime10', 'SeTime11', 'SeTime12', 'SeTime13', 'SeTime14']].values.ravel()))

print("\n")

print(pd.unique(df[['diffSeTime1', 'diffSeTime2', 'diffSeTime3', 'diffSeTime4',
       'diffSeTime5', 'diffSeTime6', 'diffSeTime7', 'diffSeTime8',
       'diffSeTime9', 'diffSeTime10', 'diffSeTime11', 'diffSeTime12',
       'diffSeTime13', 'diffSeTime14']].values.ravel()))


# We can see in columns from diffSeTime1 to diffSeTime12 very little values are found, So we have to drop them for further process.

# In[95]:


# Dropping Unnecessary columns
data=df.drop(['diffSeTime1', 'diffSeTime2', 'diffSeTime3', 'diffSeTime4',
       'diffSeTime5', 'diffSeTime6', 'diffSeTime7', 'diffSeTime8',
       'diffSeTime9', 'diffSeTime10', 'diffSeTime11', 'diffSeTime12',
       'diffSeTime13', 'diffSeTime14'],axis =1 )


# In[96]:


# Checking shape again
data.shape


# Now we have 12250 rows and 27 columns are present in the dataset.

# # Exploratory Data Analysis(Distribution)

# In[37]:


data.columns


# In[38]:


# climbRate
sns.distplot(df['climbRate'],kde=True)


# climbRate column is looking normally distributed in the range from -1000 to 1000.

# In[ ]:





# In[39]:


# Sgz
sns.distplot(df['Sgz'],kde=True)


# Sgz column is looking normally distributed and in range from -100 to 100.

# In[ ]:





# In[42]:


# p
sns.distplot(df['p'],kde=True)


# p column is showing somehow normally distribution and in ranges from -1.0 to 1.0.

# In[ ]:





# In[43]:


# q
sns.distplot(df['q'],kde=True)


# q column is showing normal distribution within the range from -0.4 to 0.4(maximum values.)

# In[ ]:





# In[44]:


# curPitch
sns.distplot(df['curPitch'],kde=True)


# curPitch column is looking somehow normally distributed and in range from -0.5.

# In[ ]:





# In[45]:


# curRoll
sns.distplot(df['curRoll'],kde=True)


# curRoll column is with less values and looking normal and in ranges from -3 to 3.

# In[ ]:





# In[46]:


# absRoll
sns.distplot(df['absRoll'],kde=True)


# absRoll column is varying and and values are negative.

# In[ ]:





# In[49]:


# diffClb
sns.distplot(df['diffClb'],kde=True)


# diffClb column is looking normally distributed and  in ranges from -40 to 40.

# In[ ]:





# In[54]:


# diffRollRate
sns.distplot(df['diffRollRate'],kde=True)


# diffRollRate column is looking somehow normally distributed.

# In[ ]:





# In[53]:


# diffDiffClb
sns.distplot(df['diffDiffClb'],kde=True)


# In diffDiffClb column most of the values are from -5 to 5.

# In[ ]:





# In[55]:


# SeTime1
sns.distplot(df['SeTime1'],kde=True)


# SeTime1 column is not looking normally distributed it is right skewed.

# In[ ]:





# In[56]:


# SeTime2
sns.distplot(df['SeTime2'],kde=True)


# SeTime2 column is not looking normally distributed it is right skewed

# In[ ]:





# In[57]:


# SeTime3
sns.distplot(df['SeTime3'],kde=True)


# SeTime3 column is not looking normally distributed it is right skewed.

# In[ ]:





# In[58]:


# SeTime4
sns.distplot(df['SeTime4'],kde=True)


# SeTime4 column is not looking normally distributed it is right skewed.

# In[ ]:





# In[59]:


# SeTime5
sns.distplot(df['SeTime5'],kde=True)


# SeTime5 column is not looking normally distributed it is right skewed.

# In[ ]:





# In[60]:


#SeTime6
sns.distplot(df['SeTime6'],kde=True)


# SeTime6 column is not looking normally distributed it is right skewed.

# In[ ]:





# In[61]:


#SeTime7
sns.distplot(df['SeTime7'],kde=True)


# SeTime7 column is not looking normally distributed it is right skewed.

# In[ ]:





# In[62]:


#SeTime8
sns.distplot(df['SeTime8'],kde=True)


# SeTime8 column is not looking normally distributed it is right skewed.

# In[ ]:





# In[63]:


#SeTime9
sns.distplot(df['SeTime9'],kde=True)


# SeTime9 column is not looking normally distributed it is right skewed.

# In[ ]:





# In[64]:


#SeTime10
sns.distplot(df['SeTime10'],kde=True)


# SeTime10 column is not looking normally distributed it is right skewed.

# In[ ]:





# In[65]:


#SeTime11
sns.distplot(df['SeTime11'],kde=True)


# SeTime11 column is not looking normally distributed it is right skewed.

# In[ ]:





# In[66]:


#SeTime12
sns.distplot(df['SeTime12'],kde=True)


# SeTime12 column is not looking normally distributed it is right skewed.

# In[ ]:





# In[67]:


#SeTime13
sns.distplot(df['SeTime13'],kde=True)


# SeTime13 column is not looking normally distributed it is right skewed.

# In[ ]:





# In[68]:


#SeTime14
sns.distplot(df['SeTime14'],kde=True)


# SeTime14 column is not looking normally distributed it is right skewed.

# In[ ]:





# In[69]:


#alpha
sns.distplot(df['alpha'],kde=True)


# alpha column is showing variation from 0.25 to 1.25.

# In[ ]:





# In[70]:


#Se
sns.distplot(df['Se'],kde=True)


# Se column is not looking normally ditributed it is bit right skewed.

# In[ ]:





# In[71]:


#goal
sns.distplot(df['goal'],kde=True)


# goal column is the target column.

# During EDA I found that in most of the columns some of the values are negative. So, I will replace it later with 0 if prediction will not give better result otherwise not.

# In[76]:


# Replacing negative value with 0
data[data < 0] = 0


# In[77]:


data.head(50)


# Now we successfully filter out the negative values.

# # Outlier detection and removal

# In[97]:


# checking outliers
data.plot(kind='box', subplots=True, layout=(5,8), figsize=(15,10))


# Outlier present in most of the columns.

# In[98]:


# removing outliers
from scipy.stats import zscore
z=np.abs(zscore(df))
threshold=3
np.where(z>3)


# In[99]:


df_new=data[(z<3).all(axis=1)]


# In[100]:


# Chceking null values
df_new.isnull().sum()


# In[101]:


# Checking Shape
df_new.shape


# So, 3346 rows has been removed as a outlier.

# # Statistical Summary

# In[102]:


df_new.describe()


# Number of count is same in every columns.
# 
# There is not much difference between mean and median(50%) in any column.
# 
# There is some difference between 75% and max in some of the columns but not much.
# 
# 

# # Correlation Matrix

# In[107]:


df.corr()


# In[106]:


plt.figure(figsize=(50,30))
sns.heatmap(df_new.corr(),annot=True,linewidths =3,fmt='0.2f',cmap="BrBG",annot_kws={"size":30}, linecolor='Black')


# Target column goal is positively correlated with absRoll and most of the columns are negatively correlated.
# climbRate is negatively correlated with curPitch.
# q is negatively correlated with diffClb.

# In[108]:


# checking the columns which are postively and negatively correlated with target column goal
plt.figure(figsize=(22,7))
df_new.corr()['goal'].sort_values(ascending=False).drop(['goal']).plot(kind='bar')
plt.xlabel('Feature',fontsize=14)
plt.ylabel('column with target names',fontsize=14)
plt.title('correlation',fontsize=20)
plt.show()


# We can see most of the columns are negatively correlated.

# # Seprating the columns into feature and target

# In[110]:


x=df_new.drop(['goal'],axis=1)
y=df_new['goal']


# In[111]:


x.shape


# In[112]:


y.shape


# # Train Test Split

# In[114]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=.25, random_state=47)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Model Building

# In[115]:


# importing libraries
from sklearn.linear_model import LinearRegression ,Lasso ,Ridge ,ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_score


# In[117]:


#for Linear Regression

lr=LinearRegression()
lr.fit(x_train,y_train)
pred_y=lr.predict(x_test)
print('r2 score',r2_score(y_test,pred_y))

print('error')
print('mean absolute error', mean_absolute_error(y_test,pred_y))
print('mean squared error', mean_squared_error(y_test,pred_y))

print('root mean squared error', np.sqrt(mean_squared_error(y_test,pred_y)))


# In[118]:


# For Linear Regression (cross validation)


score=cross_val_score(lr,x,y,cv=10)
print('cv score',np.mean(score))


# So Linear Regreesion is giving very less errors and 81% r2 score and 78% cross validation score.

# In[119]:


# Linear Regression
# graph showing the performance of the model
plt.figure(figsize=(8,6))
plt.scatter(x=y_test, y=pred_y, color='b')
plt.plot(y_test,y_test, color='r')
plt.xlabel('actual',fontsize=14)
plt.ylabel('predicted',fontsize=18)
plt.show()


# Best Fit Line is covering most of the data points.

# In[ ]:





# In[ ]:





# In[120]:


# for Lasso

ls=Lasso()

ls.fit(x_train,y_train)
pred_y=ls.predict(x_test)
print('r2 score',r2_score(y_test,pred_y))

print('error')
print('mean absolute error', mean_absolute_error(y_test,pred_y))
print('mean squared error', mean_squared_error(y_test,pred_y))

print('root mean squared error', np.sqrt(mean_squared_error(y_test,pred_y)))


# In[121]:


# For Lasso (cross validation)


score=cross_val_score(ls,x,y,cv=10)
print('cv score',np.mean(score))


# So errors are very less from Lasso Regularization but r2 score and cross validation score are not good.

# In[123]:


# Lasso
# graph showing the performance of the model
plt.figure(figsize=(8,6))
plt.scatter(x=y_test, y=pred_y, color='b')
plt.plot(y_test,y_test, color='r')
plt.xlabel('actual',fontsize=14)
plt.ylabel('predicted',fontsize=18)
plt.show()


# Best fit line is covering most of the data points.

# In[ ]:





# In[ ]:





# In[124]:


#For Ridge

rd=Ridge()

rd.fit(x_train,y_train)
pred_y=rd.predict(x_test)
print('r2 score',r2_score(y_test,pred_y))

print('error')
print('mean absolute error', mean_absolute_error(y_test,pred_y))
print('mean squared error', mean_squared_error(y_test,pred_y))

print('root mean squared error', np.sqrt(mean_squared_error(y_test,pred_y)))


# In[125]:


# For Ridge (cross  validation)


score=cross_val_score(rd,x,y,cv=10)
print('cv score',np.mean(score))


# From Ridge Regularization, errors are very less and r2 score is 80% and cv score is 78%.

# In[127]:


# ridge
# graph showing the performance of the model
plt.figure(figsize=(8,6))
plt.scatter(x=y_test, y=pred_y, color='b')
plt.plot(y_test,y_test, color='r')
plt.xlabel('actual',fontsize=14)
plt.ylabel('predicted',fontsize=18)
plt.show()


# Best Fit line is covering most of the datapoints.

# In[ ]:





# In[ ]:





# In[128]:


#For ElasticNet



en=ElasticNet()

en.fit(x_train,y_train)
pred_y=en.predict(x_test)
print('r2 score',r2_score(y_test,pred_y))

print('error')
print('mean absolute error', mean_absolute_error(y_test,pred_y))
print('mean squared error', mean_squared_error(y_test,pred_y))

print('root mean squared error', np.sqrt(mean_squared_error(y_test,pred_y)))


# In[130]:


#For ElasticNet (cross validation)

score=cross_val_score(en,x,y,cv=10)
print('cv score',np.mean(score))


# Erros are less and r2 score and cross validation scores are not good.

# In[131]:


# elasicnet
# graph showing the performance of the model
plt.figure(figsize=(8,6))
plt.scatter(x=y_test, y=pred_y, color='b')
plt.plot(y_test,y_test, color='r')
plt.xlabel('actual',fontsize=14)
plt.ylabel('predicted',fontsize=18)
plt.show()


# Best fit Line is covering most of the data points.

# So, from the training and predcitions from above algorithms we can say that our every model is performing well because errors are very less from MAE, MSE and RMSE, now will do hyper parameter tunning for better prediction.

# # Hyper Parameter Tuning using Grid Search CV

# In[132]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


# In[133]:


parameters={'criterion':['mse','mae'],'max_features':['auto','sqrt','log2']}
rf=RandomForestRegressor()
clf=GridSearchCV(rf,parameters)
clf.fit(x_train,y_train)

print(clf.best_params_)


# In[134]:


rf=RandomForestRegressor(criterion='mse',max_features='auto')
rf.fit(x_train,y_train)
rf.score(x_train,y_train)
pred_decision=rf.predict(x_test)
print('root mean squared error', np.sqrt(mean_squared_error(y_test,pred_decision)))


# # Model Saving

# In[135]:


import pickle
filename='ailerons.pkl'
pickle.dump(rf, open(filename, 'wb'))


# In[ ]:





# # Conclusion

# So after EDA, Preprocessing, Outlier Removal, Model training and building and last after prediction we are getting very less error and we can say that our model is performing extremly well and we can deploy it for future.

# In[ ]:




