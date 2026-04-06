#import libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#load the csv file

df = pd.read_csv('titanic.csv')[['Survived','Name','Pclass','Age','Sex','SibSp','Parch','Fare','Embarked']]

#feature engineering
df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)

df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr',
                                  'Major','Rev','Sir','Jonkheer','Dona'], 'Rare')

df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Family_Size'] =df['SibSp'] + df['Parch'] + 1
df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)
    
df['Age_Group'] = pd.cut(df['Age'], bins=[0,12,20,40,60,80], labels=False)

df = df.drop(['Name','SibSp',"Parch"],axis=1)

#handling catogerical data
df = pd.get_dummies(df, drop_first=True)

#features and target for the model
X = df.drop('Survived',axis=1)
y = df['Survived']

#train test split 80% training and 20% testing
X_train,X_test,y_train,y_test = train_test_split(X, y , test_size=0.2,random_state=42)

#creating the model with tuning hyperparameters
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

#train the model
rf.fit(X_train,y_train)

#make the predictions
y_pred = rf.predict(X_test)

#check the accuracy score
rf_accuracy = accuracy_score(y_test,y_pred)
print(rf_accuracy)

import joblib

joblib.dump(rf,"titanic_model.pkl")
# After training, save column order
model_columns = X.columns
import joblib
joblib.dump(model_columns, "model_columns.pkl")