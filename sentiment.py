# -*- coding: utf-8 -*-


import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('D:\ML\Stock-Sentiment-Analysis\Data.csv',encoding = 'ISO-8859-1')

train = df[df.Date < '20150101'] 
test = df[df.Date > '20141231']

data = train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True,inplace=True)

# Renaming Columns
list1 = [i for i in range(25)]
new_indx = [str(i) for i in list1]
data.columns = new_indx

# Lower case
for i in new_indx:
    data[i] = data[i].str.lower()
    
headlines = []
for row in range(len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

cv = CountVectorizer(ngram_range=(2,3))
train_data = cv.fit_transform(headlines)

pickle.dump(cv,open('transform.pkl','wb'))

r_clf = RandomForestClassifier(n_estimators=200,criterion='entropy')
r_clf.fit(train_data,train['Label'])

test_transform = []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_data = cv.transform(test_transform)
pred = r_clf.predict(test_data)
test_data.shape

r_clf.score(test_data,test['Label'])
from sklearn.metrics import accuracy_score

score = accuracy_score(test['Label'],pred)
print('Accuracy :',score,'\n')


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(train_data,train['Label'])
clf.score(test_data,test['Label']) 

filename1 = 'random_forest.pkl'
filename2 = 'naive_bayes.pkl'
#pickle.dump(r_clf,open(filename1,"wb"))
#pickle.dump(clf,open(filename2,"wb"))

pickle_out = open('random_forest.pkl','wb')
pickle.dump(r_clf,pickle_out)
pickle_out.close()
#with open(filename1,"wb") as f1:
#    pickle.dump(r_clf,f1)
#with open(filename2,"wb") as f2:
#    pickle.dump(clf,f2)