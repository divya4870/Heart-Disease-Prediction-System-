
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix

"""Data collection and processing"""

# loading csv data to pandas frame
heart_data = pd.read_csv('/content/heart.csv')

# print first rows of the dataset
heart_data.head(3)

# print last 5 rows in the dataset
heart_data.tail(2)

# checking the number of rows and columns in the dataset
heart_data.shape

# getting some info of dataset
heart_data.info()

#checking null values from dataset
heart_data.isnull().sum()

# checking the distribution of the target variable
heart_data['target'].value_counts()

# statisticall measures of the dataset
heart_data.describe()

"""spliting the features and target"""

X=heart_data.drop(columns='target',axis=1)
Y=heart_data['target']

print(X)

print(Y)

"""spliting the data into train and test data"""

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,stratify=Y, random_state=2)

print(X.shape,X_train.shape,X_test.shape )

"""model training

logisticregression
"""

model=LogisticRegression()

# training the logistic regression model with training data
model.fit(X_train,Y_train)

"""model evaluation

Acurracy score
"""

# accuracy on taring data
X_train_prediction = model.predict(X_train)
training_data_accuracy= accuracy_score(X_train_prediction,Y_train)

print('accuracy on training data : ',training_data_accuracy)

"""building a predictive system"""

input_data = (63,1,3,145,233,1,0,150,0,2.3,0,0,1)
# change the input data as numpy array
input_data_as_numpy_array= np.asarray(input_data)

#reshape the numpy array as we are prediction for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
  print("The person does not have a heart disease")
else:
    print("The person havining heart disease ")

confusion_matrix(X_train_prediction,Y_train)
