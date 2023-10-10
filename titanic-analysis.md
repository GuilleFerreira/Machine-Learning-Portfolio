---
layout: default
---

# Titanic - Data analysis with Python

We will process messy data and build a ML model to predict the survival of each passenger aboard the Titanic.
This was made following: [Titanic: Guide with sklearn and EDA]([https://github.com/GuilleFerreira](https://www.kaggle.com/code/samsonqian/titanic-guide-with-sklearn-and-eda?scriptVersionId=17744221)).


* * *

## Context
The RMS Titanic was a British passenger liner that embarked on its maiden voyage from Southampton, England, to New York City, USA, in April 1912. Regarded as one of the most luxurious and technologically advanced ships of its time, the Titanic was considered "unsinkable." However, tragedy struck when the ship struck an iceberg and sank in the North Atlantic Ocean on April 15, 1912.

The sinking of the Titanic is one of the most infamous maritime disasters in history. The ship carried over 2,200 passengers and crew, but due to a shortage of lifeboats and other factors, more than 1,500 people lost their lives in the disaster. The sinking prompted significant changes in maritime safety regulations and practices.

## Objective 
Learn data visualization, analysis and Machine Learning using Python.

## Importing Libraries and Packages

```
import numpy as np 
import pandas as pd 
import warnings
import seaborn as sns
import os 
from matplotlib import pyplot as plt

sns.set_style("whitegrid")
warnings.filterwarnings("ignore")
```

## Loading and Viewing Data Set
```
training = pd.read_csv("./input/train.csv")
testing = pd.read_csv("./input/test.csv")

print("Training Head:")
print(training.head())
print("Testing Head:")
print(testing.head())

print(training.keys())
print(testing.keys())

types_train = training.dtypes
num_values = types_train[(types_train == float)]

print("These are the numerical features:")
print(num_values)

print("Training Describe:")
print(training.describe())
```

Result:
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-analysis/paso2_img1.png?raw=true)
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-analysis/paso2_img2.png?raw=true)


## Dealing with NaN Values
```
def null_table(training, testing):
    print("Training Data Frame")
    print(pd.isnull(training).sum()) 
    print(" ")
    print("Testing Data Frame")
    print(pd.isnull(testing).sum())

null_table(training, testing)

training.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)
testing.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)

null_table(training, testing)

copy = training.copy()
copy.dropna(inplace = True)
sns.distplot(copy["Age"])

#the median will be an acceptable value to place in the NaN cells
training["Age"].fillna(training["Age"].median(), inplace = True)
testing["Age"].fillna(testing["Age"].median(), inplace = True) 
training["Embarked"].fillna("S", inplace = True)
testing["Fare"].fillna(testing["Fare"].median(), inplace = True)

null_table(training, testing)

print("Training Head:")
print(training.head())

print("Testing Head:")
print(testing.head())
```

Results:
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-analysis/paso3_img1.png?raw=true)
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-analysis/paso3_img2.png?raw=true)

## Plotting and Visualizing Data
```
sns.barplot(x="Sex", y="Survived", data=training)
plt.title("Distribution of Survival based on Gender")
plt.show()

total_survived_females = training[training.Sex == "female"]["Survived"].sum()
total_survived_males = training[training.Sex == "male"]["Survived"].sum()

print("Total people survived is: " + str((total_survived_females + total_survived_males)))
print("Proportion of Females who survived:") 
print(total_survived_females/(total_survived_females + total_survived_males))
print("Proportion of Males who survived:")
print(total_survived_males/(total_survived_females + total_survived_males))

sns.barplot(x="Pclass", y="Survived", data=training)
plt.ylabel("Survival Rate")
plt.title("Distribution of Survival Based on Class")
plt.show()

total_survived_one = training[training.Pclass == 1]["Survived"].sum()
total_survived_two = training[training.Pclass == 2]["Survived"].sum()
total_survived_three = training[training.Pclass == 3]["Survived"].sum()
total_survived_class = total_survived_one + total_survived_two + total_survived_three

print("Total people survived is: " + str(total_survived_class))
print("Proportion of Class 1 Passengers who survived:") 
print(total_survived_one/total_survived_class)
print("Proportion of Class 2 Passengers who survived:")
print(total_survived_two/total_survived_class)
print("Proportion of Class 3 Passengers who survived:")
print(total_survived_three/total_survived_class)


sns.barplot(x="Pclass", y="Survived", hue="Sex", data=training)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")
plt.show()

sns.barplot(x="Sex", y="Survived", hue="Pclass", data=training)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")
plt.show()

survived_ages = training[training.Survived == 1]["Age"]
not_survived_ages = training[training.Survived == 0]["Age"]
plt.subplot(1, 2, 1)
sns.distplot(survived_ages, kde=False)
plt.axis([0, 100, 0, 100])
plt.title("Survived")
plt.ylabel("Proportion")
plt.subplot(1, 2, 2)
sns.distplot(not_survived_ages, kde=False)
plt.axis([0, 100, 0, 100])
plt.title("Didn't Survive")
plt.subplots_adjust(right=1.7)
plt.show()

sns.stripplot(x="Survived", y="Age", data=training, jitter=True)
sns.pairplot(training)
plt.show()
```

Results:
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-analysis/paso4_img1.png?raw=true)
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-analysis/paso4_img2.png?raw=true)
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-analysis/paso4_img3.png?raw=true)
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-analysis/paso4_img4.png?raw=true)
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-analysis/paso4_img5.png?raw=true)
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-analysis/paso4_img6.png?raw=true)
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-analysis/paso4_img7.png?raw=true)

## Feature Engineering
```
print("Training Sample Antes:")
print(training.sample(5))

print("Testing Sample Antes:")
print(testing.sample(5))

set(training["Embarked"])

from sklearn.preprocessing import LabelEncoder

le_sex = LabelEncoder()
le_sex.fit(training["Sex"])

encoded_sex_training = le_sex.transform(training["Sex"])
training["Sex"] = encoded_sex_training
encoded_sex_testing = le_sex.transform(testing["Sex"])
testing["Sex"] = encoded_sex_testing

le_embarked = LabelEncoder()
le_embarked.fit(training["Embarked"])

encoded_embarked_training = le_embarked.transform(training["Embarked"])
training["Embarked"] = encoded_embarked_training
encoded_embarked_testing = le_embarked.transform(testing["Embarked"])
testing["Embarked"] = encoded_embarked_testing

print("Training Sample Después:")
print(training.sample(5))

print("Testing Sample Después:")
print(testing.sample(5))

training["FamSize"] = training["SibSp"] + training["Parch"] + 1
testing["FamSize"] = testing["SibSp"] + testing["Parch"] + 1


training["IsAlone"] = training.FamSize.apply(lambda x: 1 if x == 1 else 0)
testing["IsAlone"] = testing.FamSize.apply(lambda x: 1 if x == 1 else 0)


for name in training["Name"]:
    training["Title"] = training["Name"].str.extract("([A-Za-z]+)\.",expand=True)
    
for name in testing["Name"]:
    testing["Title"] = testing["Name"].str.extract("([A-Za-z]+)\.",expand=True)
    
print("Training Head:")
print(training.head()) #Title column added

titles = set(training["Title"]) #making it a set gets rid of all duplicates
print(titles)

title_list = list(training["Title"])
frequency_titles = []

for i in titles:
    frequency_titles.append(title_list.count(i))
    
print(frequency_titles)

titles = list(titles)

title_dataframe = pd.DataFrame({
    "Titles" : titles,
    "Frequency" : frequency_titles
})

print(title_dataframe)

title_replacements = {"Mlle": "Other", "Major": "Other", "Col": "Other", "Sir": "Other", "Don": "Other", "Mme": "Other",
          "Jonkheer": "Other", "Lady": "Other", "Capt": "Other", "Countess": "Other", "Ms": "Other", "Dona": "Other"}

training.replace({"Title": title_replacements}, inplace=True)
testing.replace({"Title": title_replacements}, inplace=True)

le_title = LabelEncoder()
le_title.fit(training["Title"])

encoded_title_training = le_title.transform(training["Title"])
training["Title"] = encoded_title_training
encoded_title_testing = le_title.transform(testing["Title"])
testing["Title"] = encoded_title_testing

training.drop("Name", axis = 1, inplace = True)
testing.drop("Name", axis = 1, inplace = True)

print("Training Sample Final:")
print(training.sample(5))

print("Training Sample Final:")
print(testing.sample(5))
```
Results:
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/titanic-analysis/paso5_img1.png?raw=true)

[Project Repo](https://github.com/GuilleFerreira).

[Go back to my projects](./projects.html).
