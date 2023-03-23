#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[3]:


data_fake = pd.read_csv("Fake2.csv")
data_true = pd.read_csv("True1.csv")


data_fake['fake'] = '1'
data_true['fake'] = '0'

data = data_fake.append(data_true,ignore_index = True)
data = data.sample(frac=1).reset_index(drop=True)
data.index = [i for i in range(0,data.shape[0])]


# In[4]:


data


# In[5]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer

X = data.copy()

score = []
kfold = StratifiedKFold(n_splits=5)
vc = CountVectorizer()
df_count = vc.fit_transform(data.text.values)


# In[6]:


print(df_count.shape)


# In[8]:


#sparse vector of the first text
print(df_count[0,:])

#print(vc.vocabulary_)
print(vc.get_feature_names_out())


# In[9]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer

X = data.copy()

score = []
kfold = StratifiedKFold(n_splits=5)

for train, valid in kfold.split(X, X.fake):
    X_train = X.loc[train]
    X_valid = X.loc[valid]
    y_train = X_train.pop("fake")
    y_valid = X_valid.pop("fake")
    
    vc = CountVectorizer()
    X_count_train = vc.fit_transform(X_train.text.values)
    X_count_valid = vc.transform(X_valid.text.values)
    
    nb = MultinomialNB()
    nb.fit(X_count_train, y_train)
    
    score.append(nb.score(X_count_valid, y_valid))
    print(nb.score(X_count_valid, y_valid))

vc = CountVectorizer()
df_count = vc.fit_transform(data.text.values)

#np.append(df_count, df[pd.get_dummies(df["subject"]).columns.values].values)
#df_count.shape

print("=====================================================")
print(np.mean(score))
nb = MultinomialNB()
nb.fit(df_count, data.fake)


# In[10]:


def predict(X):
    X_count = vc.transform(X.values)
    return nb.predict(X_count)
def predict_proba(X):
    X_count = vc.transform(X.values)
    #np.append(df_count, df[pd.get_dummies(df["subject"]).columns.values].values)
    return nb.predict_proba(X_count)


# In[11]:


demo = """



"""
prediction = predict(pd.DataFrame([demo], columns=["f"]).f)
print(prediction)
prediction_proba = predict_proba(pd.DataFrame([demo], columns=["f"]).f)
print(prediction_proba)
print("\nPrediction:","Fake" if prediction[0] == '1' else "True")
print("Probability:")
print("\tFake:",prediction_proba[0][1])
print("\tTrue:",prediction_proba[0][0])


# In[ ]:




