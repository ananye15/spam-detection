import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('spam.csv',encoding = "ISO-8859-1")

#removing unnecessary columns
dataset=dataset.iloc[:,:2]



#checking number of null values in each column
pd.isnull(dataset['v1']).values.sum()
pd.isnull(dataset['v2']).values.sum()
#filling null values with empty string
w=""
dataset.fillna(w)




#assigning spam as 0 and ham as 1
dataset.loc[dataset['v1']=='spam','v1']=0
dataset.loc[dataset['v1']=='ham','v1']=1

x=dataset.iloc[:,1]
y=dataset.iloc[:,0]




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



#Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer as tf
extracted_feature=tf(min_df=1,stop_words='english',lowercase='True')
x_train_transformed=extracted_feature.fit_transform(x_train)
x_test_transformed=extracted_feature.transform(x_test)




#converting labels
y_test=y_test.astype('int')
y_train=y_train.astype('int')

print(x_train_transformed)




#Fitting the model to training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train_transformed,y_train)




#evaluation on training data
from sklearn.metrics import accuracy_score
predicted_value_in_train=classifier.predict(x_train_transformed)
accuracy_score_training_data=accuracy_score(y_train,predicted_value_in_train)
print(accuracy_score_training_data)







#evaluation on test data

predicted_value_in_test=classifier.predict(x_test_transformed)
accuracy_score_testing_data=accuracy_score(y_test,predicted_value_in_test)
print(accuracy_score_testing_data)



#building a confusion matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predicted_value_in_test )



#testing of mails

input1=['SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info']
data_feature=extracted_feature.transform(input1)
prediction=classifier.predict(data_feature)
print(prediction)

if(prediction[0]==1):
    print('ham mail')
else:
    print('spam mail')


