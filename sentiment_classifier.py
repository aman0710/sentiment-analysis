#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random

class Sentiment:
    POSITIVE = 'POSITIVE'
    NEGATIVE = 'NEGATIVE'

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()
    
    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        else:
            return Sentiment.POSITIVE

class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews
        
    def get_text(self):
        return [x.text for x in self.reviews]
    
    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]
    
    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[: len(negative)]
        self.reviews = positive_shrunk + negative
        random.shuffle(self.reviews)


# In[3]:


import json

file_name = 'books_small_10000.json'
reviews = []

with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))

        
reviews[5].text


# In[4]:


len(reviews)


# In[5]:


from sklearn.model_selection import train_test_split

training, test = train_test_split(reviews, test_size=0.33, random_state=42)

train_container = ReviewContainer(training)
test_container = ReviewContainer(test)

train_container.evenly_distribute()
test_container.evenly_distribute()


# In[6]:


len(training)


# In[7]:


print(training[0].sentiment)


# In[8]:


train_x = train_container.get_text()
train_y = train_container.get_sentiment()


# In[9]:


train_x[1]


# In[10]:


train_y[1]


# In[11]:


test_x = test_container.get_text()
test_y = test_container.get_sentiment()


# In[12]:


test_x[0]


# In[13]:


test_y[0]


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)

test_x_vectors = vectorizer.transform(test_x)

print(train_x_vectors[0].toarray())


# In[15]:


train_x_vectors


# In[16]:


test_x_vectors


# In[17]:


test_x_vectors


# In[18]:


test_x_vectors[0]


# In[19]:


from sklearn import svm


# In[20]:


clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)


# In[21]:


test_x[0]


# In[22]:


test_y[0]


# In[23]:


print(clf_svm.predict(test_x_vectors[0]))


# In[24]:


user_input = "Overall an average book i would say. Did not enjoy much"


# In[25]:


user_input_vector = vectorizer.transform([user_input])


# In[26]:


print(user_input_vector)


# In[27]:


print(clf_svm.predict(user_input_vector))


# In[28]:


print(clf_svm.predict(test_x_vectors))


# In[29]:


from sklearn.tree import DecisionTreeClassifier

clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)


# In[30]:


print(clf_dec.predict(user_input_vector[0]))


# In[31]:


from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()
clf_gnb.fit(train_x_vectors.todense(), train_y)


# In[32]:


print(clf_gnb.predict(user_input_vector.todense()))


# In[33]:


from sklearn.linear_model import LogisticRegression

clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)


# In[34]:


print(clf_log.predict(test_x_vectors[2]))


# In[35]:


# Mean Accuracy


print(clf_svm.score(test_x_vectors, test_y))
print(clf_dec.score(test_x_vectors, test_y))
print(clf_gnb.score(test_x_vectors.todense(), test_y))
print(clf_log.score(test_x_vectors, test_y))


# In[36]:


# F1 Scores

from sklearn.metrics import f1_score

print(f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))
print(f1_score(test_y, clf_dec.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))
print(f1_score(test_y, clf_gnb.predict(test_x_vectors.todense()), average=None, labels=[Sentiment.POSITIVE,Sentiment.NEGATIVE]))
print(f1_score(test_y, clf_log.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))


# In[37]:


test_y.count(Sentiment.POSITIVE)


# In[38]:


test_set = ['good read', 'bad book do not buy', 'horrible waste of time', 'Overall a mild book i would say. Did not enjoy much']
new_test = vectorizer.transform(test_set)


# In[39]:


print(clf_log.predict(new_test))


# In[40]:


from sklearn.model_selection import GridSearchCV

parameters = {'solver': ('newton-cg', 'lbfgs','liblinear','sag','saga'), 'C': (1, 2, 3, 4, 8), 'max_iter': (400, 800,1000,1600)}
log_reg = LogisticRegression()

clf = GridSearchCV(log_reg, parameters, cv=5)

clf.fit(train_x_vectors, train_y)


# In[41]:


print(clf.score(test_x_vectors, test_y))


# In[42]:


print(f1_score(test_y, clf.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))


# In[43]:


import pickle

with open('sentiment_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)


# In[44]:


with open('sentiment_classifier.pkl', 'rb') as f:
    loaded_clf = pickle.load(f)


# In[45]:


print(test_y[3])

print(loaded_clf.predict(test_x_vectors[3]))

