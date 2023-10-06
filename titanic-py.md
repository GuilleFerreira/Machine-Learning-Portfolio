---
layout: default
---

# Titanic - Prediction with Python

We will use python with a random forest algorithm to solve the titanic dataset problem.

* * *

## Context
The RMS Titanic was a British passenger liner that embarked on its maiden voyage from Southampton, England, to New York City, USA, in April 1912. Regarded as one of the most luxurious and technologically advanced ships of its time, the Titanic was considered "unsinkable." However, tragedy struck when the ship struck an iceberg and sank in the North Atlantic Ocean on April 15, 1912.
The sinking of the Titanic is one of the most infamous maritime disasters in history. The ship carried over 2,200 passengers and crew, but due to a shortage of lifeboats and other factors, more than 1,500 people lost their lives in the disaster. The sinking prompted significant changes in maritime safety regulations and practices.

## Objective 
The dataset contains the data of the Titanic passengers, The objective is to predict, based on the attributes we have, whether a passenger survives or not..
This is a binary classification problem using a supervised algorithm, in this case using random forest. 

### Attributes
* PassengerId: Passenger Id. Type Integer.
* Survived: If the passenger survived or not. Type Integer.
* Pclass: Passenger class. Integer type.
* Name: Passenger's name. String type.
* Sex: Passenger's sex. Type string.
* Age: Passenger age. Type float. 177 missing values
* SibSp: Number of passenger's siblings/spouses on board. Integer type.
* Parch: Number of parents/children of the passenger on board. Integer type.
* Ticket: Passenger's ticket number. String type.
* Fare: Passenger's ticket price. Type float.
* Cabin: Passenger's cabin. Type string. 687 missing values
* Embarked: Passenger's embarkation port. Type string. 2 missing values

## Data preparation
First we import the pandas library that we will use to load the file, load it and print information about the file and its number of missing values.
```
import pandas as pd

df = pd.read_csv('Titanic-Dataset.csv')

# Information about the dataframe
df.info()

# Check for missing values using isna() function
missing_values = df.isna()

# Count missing values in each column
missing_count = missing_values.sum()

# Print count of missing values
print("Missing value counts:")
print(missing_count)
```
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-py/info.png?raw=true)

![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-py/missing1.png?raw=true)

From this we know that the attributes Age, Cabin and Embarked have missing values. 
We now proceed to delete the attributes that we do not consider important to solve this problem.

The Cabin attribute is deleted because it has too many missing values.
```
df = df.drop('Cabin', axis=1)
```

The Name, Fare, Ticket and PassengerId attributes are eliminated because we do not consider them important on this occasion.
```
df = df.drop('Name', axis=1)
df = df.drop('Fare', axis=1)
df = df.drop('Ticket', axis=1)
df = df.drop('PassengerId', axis=1)
```

We fill in the missing values of the age attribute with the average of the existing values.
```
df['Age'] = df['Age'].fillna(df['Age'].mean())
```

We fill in the missing values in embarked with the most frequent value, and change the values from categorical to numeric using the get_dummies() function.
```
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
embarked_numerical = pd.get_dummies(df['Embarked'])
df = df.drop('Embarked',axis=1)
df = df.join(embarked_numerical)
```

In addition we convert the categorical values from Sex to Numerical, also using get_dummies()
```
sex_numerical = pd.get_dummies(df['Sex'])
df = df.drop('Sex', axis=1)
df = df.join(sex_numerical)
```

Finally we recheck that there is no missing data and print the first five rows of the dataframe to check.
```
missing_values = df.isna()
missing_count = missing_values.sum()
print("Missing value counts:")
print(missing_count)

print(df.head())
```
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-py/missing2.png?raw=true)

![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-py/head.png?raw=true)


## Create testing and training dataset
We will use sklearn train_test_split to split the dataset into 70% training and 30% testing. But first we must remove the Survived attribute because it is the one we are trying to predict, while the y variable will be assigned to survived.
```
from sklearn.model_selection import train_test_split
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

## Apply Random Forest
Finally we import RandomForestClassifier and accuracy_score from sklearn in order to initialize a random forest and train it with the training datasets, then use them to predict using the test dataset and evaluate its accuracy.
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
prediction = random_forest.predict(X_test)
print('Accuracy score =', accuracy_score(y_test, prediction))
```

## Results
After all these steps we were able to arrive at a prediction with a 77% accuracy of being able to predict whether or not a passenger survives the Titanic using this algorithm.
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-py/accuracy.png?raw=true)
