# -*- coding: utf-8 -*-
import numpy as np
import re
import pickle
import nltk
from sklearn.datasets import load_files

dataset = load_files('datasets/')
X, y = dataset.data, dataset.target

with open('X.pickle', 'wb') as f:
    pickle.dump(X, f)
with open('y.pickle', 'wb') as f:
    pickle.dump(y, f)

corpus = []
for i in range(len(X)):
    # print(X[i].decode("utf-8", errors="ignore"))
    tempData = re.sub('[,!?‘’:.।;()]', '', X[i].decode("utf-8-sig", errors="ignore"))
    tempData = re.sub('[-]', ' ', tempData)
    # tempData = re.sub('[\ufeff]', '', tempData)
    tempData = re.sub('[\u200c]', '', tempData)
    # print(tempData)
    corpus.append(tempData)
    
    
#for feature extraction using CountVectorizer and then tensform it Tfidf model
from sklearn.feature_extraction.text import CountVectorizer 
vectorize = CountVectorizer(min_df=2,max_df=0.7);
X =  vectorize.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
transforme = TfidfTransformer()
X = transforme.fit_transform(X).toarray()



# split out total dataset into trainiing and test data set
from sklearn.model_selection import train_test_split
text_train,text_test,sent_train,sent_test = train_test_split(X,y,test_size=0.2,random_state=0)


# to fit dataset in machine learning algorithm and and create a trained model 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)

# testing our model 
sent_pred = classifier.predict(text_test)

# performance measure using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test,sent_pred)

classifier.score(text_test,sent_test)

from sklearn.metrics import accuracy_score

print ("LogisticRegression accuracy score : ",accuracy_score(sent_pred,sent_test))


from sklearn.naive_bayes import MultinomialNB

nbclf = MultinomialNB()

nbclf.fit(text_train,sent_train)

sent_pred_nb = nbclf.predict(text_test)

cmNB = confusion_matrix(sent_test,sent_pred_nb)

print ("Naive Bayes accuracy score : ",accuracy_score(sent_pred_nb,sent_test))

from sklearn.ensemble import RandomForestClassifier
rfCLF = RandomForestClassifier(n_estimators=160)

rfCLF.fit(text_train,sent_train)
sent_pred_rf = rfCLF.predict(text_test)

cmRF = confusion_matrix(sent_test,sent_pred_rf)

print ("RandomForestClassifier accuracy score : ",accuracy_score(sent_pred_rf,sent_test))

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(text_train,sent_train)
sent_pred_svm = svclassifier.predict(text_test)
cmSVM = confusion_matrix(sent_test,sent_pred_svm)
print ("SVM accuracy score : ",accuracy_score(sent_pred_svm,sent_test))




































