---
layout: default
---

# Wine - Data Preprocessing with Python

This dataset contains the results of the chemical analysis of different wines grown in the same region of Italy but from different cultivars.
We have 14 attributes (including region) among which are Alcohol, Malic acid, Ash, Alkalinity of ash, etc.
The dataset was created for testing and is intended to predict the wine crop according to the wine attributes.

In this opportunity we are going to perform the work of data preparation, this is essential to ensure that the data is suitable for the construction and training of Machine Learning models.

Now we will show the python code of how we normalize and standardize using the pandas and sklearn libraries.

## Code
```
# Import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# File path and column names
filePath = 'wine.data'
columnNames = ['Class','Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline' ]

df = pd.read_csv(filePath, delimiter=',', names=columnNames)


# Print information about the DataFrame
info = df.info()
print(info)

# Print description of the DataFrame
describe = df.describe()
print("Description of the DataFrame: \n", describe, "\n")


# Print the first 5 rows of the Original DataFrame
print("First 5 rows of the Original DataFrame: \n", df.head(), "\n")


# Standardize data using StandardScaler
scaler1 = StandardScaler()
dfStandardized = pd.DataFrame(scaler1.fit_transform(df), columns=df.columns)


# Print the first 5 rows of the standardized DataFrame
print("First 5 rows of the Standardized DataFrame: \n", dfStandardized.head(), "\n")


# Normalize data using MinMaxScaler
scaler2 = MinMaxScaler()
dfNormalized = pd.DataFrame(scaler2.fit_transform(df), columns=df.columns)


# Print the first 5 rows of the normalized DataFrame
print("First 5 rows of the Normalized DataFrame: \n", dfNormalized.head(), "\n")
```

## Results
Information about the dataframe
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/wine/information.png?raw=true)

Description about the dataframe
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/wine/description.png?raw=true)

Original dataframe
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/wine/original.png?raw=true)

Standardized dataframe
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/wine/standarized.png?raw=true)

Normalized dataframe
![Octocat](https://github.com/GuilleFerreira/Machine-Learning-Portfolio/blob/main/assets/img/wine/normalized.png?raw=true)


[Go back to my projects](./projects.html).

