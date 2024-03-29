# -*- coding: utf-8 -*-
import numpy as np
import re
import pickle
from sklearn.datasets import load_files

dataset = load_files('datasets/')
X, y = dataset.data, dataset.target

with open('X.pickle', 'wb') as f:
    pickle.dump(X, f)
with open('y.pickle', 'wb') as f:
    pickle.dump(y, f)

# clean data using regular expression
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
    '''
from sklearn.feature_extraction.text import CountVectorizer 
vectorize = CountVectorizer(min_df=3,max_df=0.4);
X =  vectorize.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
transforme = TfidfTransformer()
X = transforme.fit_transform(X).toarray()

'''
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=370,min_df=3,max_df=0.6)
X = vectorizer.fit_transform(corpus).toarray()





# split out total dataset into trainiing and test data set
from sklearn.model_selection import train_test_split
text_train,text_test,label_train,label_test = train_test_split(X,y,test_size=0.2,random_state=0)


# to fit dataset in machine learning algorithm and and create a trained model 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,label_train)

# testing our model 
label_pred = classifier.predict(text_test)

# performance measure using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(label_test,label_pred)

classifier.score(text_test,label_test)

from sklearn.metrics import accuracy_score

print ("LogisticRegression accuracy score : ",accuracy_score(label_pred,label_test)*100)


from sklearn.naive_bayes import MultinomialNB

nbclf = MultinomialNB()

nbclf.fit(text_train,label_train)

label_pred_nb = nbclf.predict(text_test)

cmNB = confusion_matrix(label_test,label_pred_nb)

print ("Naive Bayes accuracy score : ",accuracy_score(label_pred_nb,label_test)*100)

from sklearn.ensemble import RandomForestClassifier
rfCLF = RandomForestClassifier(n_estimators=160)

rfCLF.fit(text_train,label_train)
label_pred_rf = rfCLF.predict(text_test)

cmRF = confusion_matrix(label_test,label_pred_rf)

print ("RandomForestClassifier accuracy score : ",accuracy_score(label_pred_rf,label_test)*100)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(text_train,label_train)
label_pred_svm = svclassifier.predict(text_test)
cmSVM = confusion_matrix(label_test,label_pred_svm)
print ("SVM accuracy score : ",accuracy_score(label_pred_svm,label_test)*100)




# Pickleing the classifier
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)

# pickleing the Tfidfmodel
    
with open('tifdf.pickle','wb') as f:
    pickle.dump(vectorizer,f)
    
with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)

with open('tifdf.pickle','rb') as f:
    tfidf = pickle.load(f)
    
sample = ['ফেসবুকের পাঁচ শ কোটি ডলার জরিমানা']

sample = tfidf.transform(sample).toarray()

print(clf.predict(sample))







































