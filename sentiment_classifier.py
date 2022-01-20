#!/usr/bin/env python
# coding: utf-8

# In[222]:


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


# In[223]:


import json

file_name = 'books_small_10000.json'
reviews = []

with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))

        
reviews[5].text


# In[224]:


len(reviews)


# In[225]:


from sklearn.model_selection import train_test_split

training, test = train_test_split(reviews, test_size=0.33, random_state=42)

train_container = ReviewContainer(training)
test_container = ReviewContainer(test)

train_container.evenly_distribute()
test_container.evenly_distribute()


# In[226]:


len(training)


# In[227]:


print(training[0].sentiment)


# In[228]:


train_x = train_container.get_text()
train_y = train_container.get_sentiment()


# In[229]:


train_x[1]


# In[230]:


train_y[1]


# In[231]:


test_x = test_container.get_text()
test_y = test_container.get_sentiment()


# In[232]:


test_x[0]


# In[233]:


test_y[0]


# In[234]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)

test_x_vectors = vectorizer.transform(test_x)

print(train_x_vectors[0].toarray())


# In[235]:


train_x_vectors


# In[236]:


test_x_vectors


# In[237]:


test_x_vectors


# In[238]:


test_x_vectors[0]


# In[239]:


from sklearn import svm


# In[240]:


clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)


# In[241]:


test_x[0]


# In[242]:


test_y[0]


# In[243]:


print(clf_svm.predict(test_x_vectors[0]))


# In[244]:


user_input = "Overall an average book i would say. Did not enjoy much"


# In[245]:


user_input_vector = vectorizer.transform([user_input])


# In[246]:


print(user_input_vector)


# In[247]:


print(clf_svm.predict(user_input_vector))


# In[248]:


print(clf_svm.predict(test_x_vectors))


# In[249]:


from sklearn.tree import DecisionTreeClassifier

clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)


# In[250]:


print(clf_dec.predict(user_input_vector[0]))


# In[251]:


from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()
clf_gnb.fit(train_x_vectors.todense(), train_y)


# In[252]:


print(clf_gnb.predict(user_input_vector.todense()))


# In[253]:


from sklearn.linear_model import LogisticRegression

clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)


# In[254]:


print(clf_log.predict(test_x_vectors[2]))


# In[255]:


# Mean Accuracy


print(clf_svm.score(test_x_vectors, test_y))
print(clf_dec.score(test_x_vectors, test_y))
print(clf_gnb.score(test_x_vectors.todense(), test_y))
print(clf_log.score(test_x_vectors, test_y))


# In[257]:


# F1 Scores

from sklearn.metrics import f1_score

print(f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))
print(f1_score(test_y, clf_dec.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))
print(f1_score(test_y, clf_gnb.predict(test_x_vectors.todense()), average=None, labels=[Sentiment.POSITIVE,Sentiment.NEGATIVE]))
print(f1_score(test_y, clf_log.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))


# In[262]:


test_y.count(Sentiment.POSITIVE)


# In[263]:


test_set = ['good read', 'bad book do not buy', 'horrible waste of time', 'Overall a mild book i would say. Did not enjoy much']
new_test = vectorizer.transform(test_set)


# In[264]:


print(clf_log.predict(new_test))


# In[265]:


from sklearn.model_selection import GridSearchCV

parameters = {'solver': ('newton-cg', 'lbfgs','liblinear','sag','saga'), 'C': (1, 2, 3, 4, 8), 'max_iter': (400, 800,1000,1600)}
log_reg = LogisticRegression()

clf = GridSearchCV(log_reg, parameters, cv=5)

clf.fit(train_x_vectors, train_y)


# In[266]:


print(clf.score(test_x_vectors, test_y))


# In[267]:


import pickle

with open('sentiment_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)


# In[268]:


with open('sentiment_classifier.pkl', 'rb') as f:
    loaded_clf = pickle.load(f)


# In[269]:


print(test_y[3])

print(loaded_clf.predict(test_x_vectors[3]))

