import pandas as pd
import re
from pandas import DataFrame
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import BernoulliNB

tags_list=[] #tags_list  
list_train=[] #desc_words_list
list_test=[]
Train=open('/home/palak/Machine_Learning_challenges/hacker_tagging_raw_job_description/train.tsv','r')
Test=open('/home/palak/Machine_Learning_challenges/hacker_tagging_raw_job_description/test.tsv','r')
counter=0
for line in Train :
    j=line.split('\t')
    counter+=1
    tags_list.append(j[0].split())
    list_train.append(j[1])
    #if counter==5 :
    #    break
for line in Test :
    counter+=1
    list_test.append(line)

#print(desc_list_train)

X_train=np.array(list_train)
#print(X_train.shape)
X_test=np.array(list_test)
X_train_tag=np.array(tags_list)

X_train=list_train
#print(X_train.shape)
X_test=list_test
X_train_tag=tags_list


target_names=[  "part-time-job",
                "full-time-job",
                "hourly-wage",
                "salary",
                "associate-needed",
                "bs-degree-needed",
                "ms-or-phd-needed",
                "licence-needed",
                "1-year-experience-needed",
                "2-4-years-experience-needed",
                "5-plus-years-experience-needed",
                "supervising-job" ]


stopwords=['\xef\x80\xad','\xef\x80\xad','\xe2\x80\x93','\xe2\x80\xa6','\xe2\x80\xa2','\xe2\x80\xa2','\xe2\x80\x98','\xe2\x80\x99',',',';','.','-',':',')','(','!','/','EOF','\n','+']

for i in range(1,4376):
    for word in stopwords:
        if word in X_train[i]:
            X_train[i]=X_train[i].replace(word,"")
index=[]
for i in range(1,4376):
    if X_train_tag[i]==[]:
        index.append(i)


X_train_tag = np.delete(X_train_tag, index)
X_train= np.delete(X_train, index)
#print(X_train_tag.shape)


mlb = MultiLabelBinarizer()         #########score 587            
Y = mlb.fit_transform(X_train_tag)
#print(X_train.shape)
#print(Y.shape)
classifier = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1,2))),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(AdaBoostClassifier(n_estimators=400)))])

classifier.fit(X_train, Y)
predicted = classifier.predict(X_test)
all_labels = mlb.inverse_transform(predicted)

for item, labels in zip(X_test, all_labels):
    print(' {1}'.format(item, ' '.join(labels)))

