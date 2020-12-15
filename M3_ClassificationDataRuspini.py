import pandas as pd
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

#program read file data ruspini
data = pd.read_csv('G:/A S2/SEM 1/AI_Pak Ali Ridho/M3/ruspini.csv')
print(data.head)

#program split data (80% of first data from each class label as training data)
#program split data (20% of the rest data from each class label as testing data)
train_data1=data[0:16]
test_data1=data[16:20]
train_data2=data[20:34]
test_data2=data[34:37]
train_data3=data[37:55]
test_data3=data[55:60]
train_data4=data[60:72]
test_data4=data[72:75]

#untuk melihat jumlah data (split) train1 dan test1 sesuai dengan rule
print(train_data1.shape)
print(test_data1.shape)
#untuk melihat data train1 dan test1 sesuai dengan rule
print(train_data1)
print(test_data1)

#untuk melihat jumlah data (split) train2 dan test2 sesuai dengan rule
print(train_data2.shape)
print(test_data2.shape)
#untuk melihat data train1 dan test1 sesuai dengan rule
print(train_data2)
print(test_data2)

#untuk melihat jumlah data (split) train3 dan test3 sesuai dengan rule
print(train_data3.shape)
print(test_data3.shape)
#untuk melihat data train1 dan test1 sesuai dengan rule
print(train_data3)
print(test_data3)

#untuk melihat jumlah data (split) train4 dan test4 sesuai dengan rule
print(train_data4.shape)
print(test_data4.shape)
#untuk melihat data train1 dan test1 sesuai dengan rule
print(train_data4)
print(test_data4)

#program Concatenate train_data and test_data
train=train_data1.append([train_data2,train_data3,train_data4])
print(train)
test=test_data1.append([test_data2,test_data3,test_data4])


#program shuffle the data to erase order dependent anomaly in data
train=shuffle(train)
print(train)
test=shuffle(test)

#program removing label from train and test input
train_x = train.drop(columns=['CLASS','#'])
test_x  = test.drop(columns=['CLASS','#'])

#program making class label as output
train_y = train['CLASS']
test_y = test['CLASS']
#input
print(train_x)
#output machine learning
print(train_y)

#program to KNN Classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_x, train_y)

#program to predict test data using KNN
pred_y = classifier.predict(test_x)
print(classification_report(test_y, pred_y))
print(pred_y)
print(test_y)
