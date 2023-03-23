#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier


# In[3]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.offline import iplot


# In[4]:


import pandas_profiling as pp


# In[6]:


pp.ProfileReport(fake_df)


# In[7]:



fake_df = pd.read_csv('Fake2.csv')
fake_df['label'] = 0


# In[8]:


fake_df.shape


# In[9]:


True1_df = pd.read_csv('True1.csv')
True1_df['label'] = 1


# In[10]:



pp.ProfileReport(True1_df)


# In[11]:


True1_df.shape


# In[12]:


df = True1_df.copy(deep=True)
df = df.append(fake_df, ignore_index=True)
df


# In[12]:


df.shape


# In[13]:


print(f"Dataset subject unique values: {df['subject'].unique()}")


# In[14]:


print(df.columns[df.isnull().any()])


# In[15]:


sns.countplot(x=df['label'], data=df)


# In[16]:


fake_text = ' '.join(fake_df['title']) + ' '.join(fake_df['text'])
True1_text = ' '.join(True1_df['title']) + ' '.join(True1_df['text'])

wordcloud_fake= WordCloud(stopwords=ENGLISH_STOP_WORDS,
                           background_color='white', 
                           width=1200, height=1000).generate(fake_text)
wordcloud_True1 = WordCloud(stopwords=ENGLISH_STOP_WORDS,
                           background_color='white', 
                           width=1200, height=1000).generate(True1_text)

plt.figure(figsize = [9, 9])
plt.imshow(wordcloud_fake)
plt.axis('off')
plt.title('Fake News')
plt.show()

plt.figure(figsize = [9, 9])
plt.imshow(wordcloud_True1)
plt.axis('off')
plt.title('Real News')
plt.show()


# In[17]:


#Data Pre-processing
# Concatenate titles & text
X = df['title'] + ' ' + df['text']
y = df['label']

punctuation_regex = re.compile(r'[^\w\s]+')
urls_regex = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+'
                        r'[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+['
                        r'a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-'
                        r'zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')

# Apply data cleaning
X = X.apply(lambda x: urls_regex.sub('', str(x)))
X = X.apply(lambda x: ' '.join([item for item in x.split() if item not in ENGLISH_STOP_WORDS]))
X = X.apply(lambda x: punctuation_regex.sub('', str(x)))


# In[18]:


# Split data to 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=10)


# In[19]:


# Set up the model pipeline
# Note: the parameters are extracted through offline gridsearch param tuning
# We are using TF-IDF vectorizer in order to transform the text.
pipeline = Pipeline(
    [
        ('vect', TfidfVectorizer(lowercase=True, max_features=10000, ngram_range=(1,2))),
        ('clf', RandomForestClassifier(max_features='sqrt', n_estimators=1000, n_jobs=-1))
    ]
)

# Creating a StratifiedKFold object with 5 splits
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

scores = cross_validate(pipeline, X_train, y_train,
                        scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                        cv=5,
                        n_jobs=-1,
                        return_train_score=False)

print('Cross validation scores', scores)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
disp.plot()
plt.show()


# In[ ]:





# In[ ]:




