
# coding: utf-8

# In[1]:

# Importing libraries used in analysis 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Importing Data file and printing first 5 observations
data = pd.read_csv('C:\Users\rubyk\Desktop\MSBA\MSBA\4.MSBA Spring17\Big Data Analytics\big data proj\proj_code')
dataset = pd.DataFrame(data)
dataset.head()


# In[2]:

# Listing number of rows and columns parsed in whole data set 
dataset.shape


# In[3]:

#Summary Statistics 
dataset.describe()


# In[4]:

# Sum for number of observations with missing values 
dataset.isnull().sum()


# In[5]:

# Displaying the frequency chart for output variable (bstatus = bankruptcy status)
plt.figure(figsize=(10,6))
sns.countplot(x='bstatus',data = dataset)


# In[6]:

data_preprocessing = dataset.copy()
data_preprocessing.drop(['ID', 'bstatus'], axis = 1, inplace = True)


# In[7]:

normalized_data = (data_preprocessing - data_preprocessing.mean())/data_preprocessing.std()
normalized_data.head()


# In[8]:

# Check for Correlation among all variables  
corr = normalized_data[normalized_data.columns].corr()
fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(corr,annot = True, linewidths=0.5, ax=ax)


# In[9]:

# Check for Skewness in the dataset 
normalized_data.skew()


# In[10]:

#We draw the histograms 
figure = plt.figure()
figure.add_subplot(121)   
plt.hist(normalized_data.cf_td,facecolor='blue',alpha=0.75, bins=range(-5, 5, 1)) 
plt.xlabel("cf_td") 
plt.title("cf_td Histogram")


# In[11]:

normalized_data.head()


# In[12]:

skness_positive = []
skness_negative = []
for ratio in normalized_data:
    sk = normalized_data[ratio].skew()
    if sk > 2:
        if np.min(normalized_data[ratio]) < 0:
            for value in normalized_data[ratio]:
                transformed_value = np.log10(value + np.abs(np.min(normalized_data[ratio])) + 1)
                skness_positive.append(transformed_value)
            a = pd.Series(skness_positive)
            print(ratio, a.skew())
            normalized_data.loc[:, ratio] = a
            del skness_positive[:]
        else: 
            for value in normalized_data[ratio]:
                c = np.log10(value + 1)
                skness_positive.append(c)
            a = pd.Series(skness_positive)
            print(ratio, a.skew())
            normalized_data.loc[:, ratio] = a
            del skness_positive[:]
    else:
        if np.min(normalized_data[ratio]) < 0:
            for value in normalized_data[ratio]:
                transformed_value = np.sqrt(value + np.abs(np.min(normalized_data[ratio])) + 1)
                skness_negative.append(transformed_value)
            x = pd.Series(skness_negative)
            print(ratio, x.skew())
            normalized_data.loc[:, ratio] = x
            del skness_negative[:]
        else: 
            for value in normalized_data[ratio]:
                y = np.sqrt(value + 1)
                skness_negative.append(y)
            x = pd.Series(skness_negative)
            print(ratio, x.skew())
            normalized_data.loc[:, ratio] = x
            del skness_negative[:]


# In[13]:

normalized_data.head()


# In[14]:

normalized_data.columns = ['trans_cf_td', 'trans_ca_cl', 'trans_re_ta', 'trans_ni_ta', 'trans_td_ta', 'trans_s_ta', 'trans_wc_ta', 'trans_wc_s', 'trans_c_cl', 'trans_cl_e', 'trans_in_s', 'trans_mve_td']


# In[15]:

normalized_data.head()


# In[16]:

dataset = dataset.join(normalized_data)


# In[17]:

dataset.head()


# In[29]:

dataset.to_csv('C:/MSBA/Big Data Analytics/Project/bankruptcy_data/transformed_new_data.csv', sep=',')


# In[18]:

data = dataset.copy()
data.drop(['cf_td', 'ca_cl', 're_ta', 'ni_ta', 'td_ta', 's_ta', 'wc_ta', 'wc_s', 'c_cl', 'cl_e', 'in_s', 'mve_td'], axis = 1, inplace = True)


# In[19]:

data.head()


# In[32]:

data.to_csv('C:/MSBA/Big Data Analytics/Project/bankruptcy_data/data.csv', sep=',')


# In[20]:

# Check for Correlation among all variables  
corr1 = data[data.columns].corr()
fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(corr1,annot = True, linewidths=0.5, ax=ax)


# In[21]:

train_one = pd.read_csv("C:/MSBA/Big Data Analytics/Project/bankruptcy_data/train_subset_one.csv")
train_one.head()


# In[22]:

# Displaying the frequency chart for output variable (bstatus = bankruptcy status)
plt.figure(figsize=(10,6))
sns.countplot(x='bstatus',data = train_one)


# In[23]:

from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
# create dataframes with an intercept column and dummy variables for
# occupation and occupation_husb
y, X = dmatrices('bstatus ~ trans_cf_td + trans_ca_cl + trans_re_ta + trans_ni_ta + trans_td_ta + trans_s_ta + trans_wc_ta + trans_wc_s + trans_c_cl + trans_cl_e + trans_in_s + trans_mve_td',
                  train_one, return_type="dataframe")
print (X.columns)


# In[24]:

# flatten y into a 1-D array
y = np.ravel(y)


# In[25]:

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)
# check the accuracy on the training set
model.score(X, y)


# In[26]:

# what percentage was bankrupt 
y.mean()


# In[27]:

# predict class labels for the data
predicted = model.predict(X)
print (predicted)


# In[28]:

# generate class probabilities
probs = model.predict_proba(X)
print (probs)


# In[29]:

from sklearn import metrics
# generate evaluation metrics
print (metrics.accuracy_score(y, predicted))
print (metrics.roc_auc_score(y, probs[:, 1]))


# In[30]:

import seaborn as sns
from sklearn.metrics import confusion_matrix
print (metrics.confusion_matrix(y, predicted))
print (metrics.classification_report(y, predicted))
cm = confusion_matrix(y, predicted)
sns.heatmap(cm,  
            xticklabels=['Non Bankrupt', 'Bankrupt'], 
            yticklabels=['Non Bankrupt', 'Bankrupt'])


# In[31]:

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)


# In[32]:

from sklearn.tree import DecisionTreeClassifier
train_two = pd.read_csv("C:/MSBA/Big Data Analytics/Project/bankruptcy_data/train_subset_two.csv")
train_two.head()


# In[33]:

y2, X2 = dmatrices('bstatus ~ trans_cf_td + trans_ca_cl + trans_re_ta + trans_ni_ta + trans_td_ta + trans_s_ta + trans_wc_ta + trans_wc_s + trans_c_cl + trans_cl_e + trans_in_s + trans_mve_td',
                  train_two, return_type="dataframe")
print (X2.columns)
# flatten y into a 1-D array
y2 = np.ravel(y2)


# In[34]:

# Building a decision tree on train data
classifier = DecisionTreeClassifier()
classifier = classifier.fit(X2, y2)
# check the accuracy on the training set
classifier.score(X2, y2)


# In[35]:

# predict class labels for the data
predicted_tree = classifier.predict(X2)
print (predicted_tree)


# In[36]:

# generate evaluation metrics
print (metrics.accuracy_score(y2, predicted_tree))


# In[37]:

print (metrics.confusion_matrix(y2, predicted_tree))
print (metrics.classification_report(y2, predicted_tree))
cm_tree = confusion_matrix(y2, predicted_tree)
sns.heatmap(cm_tree,  
            xticklabels=['Non Bankrupt', 'Bankrupt'], 
            yticklabels=['Non Bankrupt', 'Bankrupt'])


# In[38]:

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
print(vif)


# In[39]:

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(12,12,12))
mlp.fit(X2,y2)


# In[40]:

predict_MLP = mlp.predict(X2)
print(predict_MLP)
print (metrics.accuracy_score(y2, predict_MLP))


# In[41]:

print (metrics.confusion_matrix(y2, predict_MLP))
print (metrics.classification_report(y2, predict_MLP))
cm_MLP = confusion_matrix(y2, predict_MLP)
sns.heatmap(cm_MLP,  
            xticklabels=['Non Bankrupt', 'Bankrupt'], 
            yticklabels=['Non Bankrupt', 'Bankrupt'])


# In[42]:

from sklearn.svm import SVC
train_three = pd.read_csv("C:/MSBA/Big Data Analytics/Project/bankruptcy_data/train_subset_three.csv")
train_three.head()


# In[43]:

y3, X3 = dmatrices('bstatus ~ trans_cf_td + trans_ca_cl + trans_re_ta + trans_ni_ta + trans_td_ta + trans_s_ta + trans_wc_ta + trans_wc_s + trans_c_cl + trans_cl_e + trans_in_s + trans_mve_td',
                  train_three, return_type="dataframe")
print (X3.columns)
# flatten y into a 1-D array
y3 = np.ravel(y3)


# In[44]:

# Building a Support Vector Machine on train data
SVC = SVC()
SVC = SVC.fit(X3, y3)
# check the accuracy on the training set
SVC.score(X3, y3)


# In[45]:

# predict class labels for the data
predicted_SVC = SVC.predict(X3)
print (predicted_SVC)


# In[46]:

# generate evaluation metrics
print (metrics.accuracy_score(y3, predicted_SVC))


# In[47]:

print (metrics.confusion_matrix(y3, predicted_SVC))
print (metrics.classification_report(y3, predicted_SVC))
cm_SVC = confusion_matrix(y3, predicted_SVC)
sns.heatmap(cm_SVC,  
            xticklabels=['Non Bankrupt', 'Bankrupt'], 
            yticklabels=['Non Bankrupt', 'Bankrupt'])


# In[48]:

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X3.values, i) for i in range(X3.shape[1])]
print(vif)


# In[49]:

test_data = pd.read_csv("C:/MSBA/Big Data Analytics/Project/bankruptcy_data/test_data_new.csv")
test_data.head()


# In[50]:

# Displaying the frequency chart for output variable (bstatus = bankruptcy status)
plt.figure(figsize=(10,6))
sns.countplot(x='bstatus',data = test_data)


# In[51]:

y_test, X_test = dmatrices('bstatus ~ trans_cf_td + trans_ca_cl + trans_re_ta + trans_ni_ta + trans_td_ta + trans_s_ta + trans_wc_ta + trans_wc_s + trans_c_cl + trans_cl_e + trans_in_s + trans_mve_td',
                  test_data, return_type="dataframe")
print (X_test.columns)
# flatten y into a 1-D array
y_test = np.ravel(y_test)


# In[52]:

predicted_logit = model.predict(X_test)
print (predicted_logit)


# In[53]:

model.score(X_test, y_test)


# In[54]:

p_lm = pd.Series(predicted_logit)
p_lm.columns = ['p_lm']
print(p_lm)


# In[60]:

predicted_MLP = mlp.predict(X_test)
print (predicted_MLP)
mlp.score(X_test, y_test)
p_MLP = pd.Series(predicted_MLP)
p_MLP.columns = ['p_MLP']
print(p_MLP)


# In[56]:

predicted_SVC = SVC.predict(X_test)
print (predicted_SVC)
SVC.score(X_test, y_test)
p_SVC = pd.Series(predicted_SVC)
p_SVC.columns = ['p_SVC']
print(p_SVC)


# In[61]:

final_pred = pd.concat([p_lm, p_MLP, p_SVC], axis =1)
final_pred.columns = ['p_lm', 'p_MLP', 'p_SVC']
print (final_pred)


# In[62]:

predicted_bankrupt = final_pred.mode(axis=1)
print(predicted_bankrupt)


# In[63]:

# generate evaluation metrics
print (metrics.accuracy_score(y_test, predicted_logit))
print (metrics.accuracy_score(y_test, predicted_MLP))
print (metrics.accuracy_score(y_test, predicted_SVC))
print (metrics.accuracy_score(y_test, predicted_bankrupt))


# In[64]:

print (metrics.confusion_matrix(y_test, predicted_bankrupt))
print (metrics.classification_report(y_test, predicted_bankrupt))
cm_test = confusion_matrix(y_test, predicted_bankrupt)
sns.heatmap(cm_test,  
            xticklabels=['Non Bankrupt', 'Bankrupt'], 
            yticklabels=['Non Bankrupt', 'Bankrupt'])


# In[65]:

y_test.mean()


# In[66]:

# calculate the fpr and tpr for all thresholds of the classification
probs = mlp.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:



