import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load Data
url = r"https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)


# Manual Statistics
age = df["Age"].dropna().to_numpy()

mean_age = np.sum(age) / len(age)

differences = age - mean_age
squared_differences = differences ** 2
variance_age = np.sum(squared_differences) / len(age)

std_age = np.sqrt(variance_age)

#Covariance matrix


#drop ids, no use
df = df.drop(columns=['PassengerId',"Cabin"])
df_numeric = df.select_dtypes(include='number').dropna()
cov_matrix = np.cov(df_numeric,rowvar=False)

#correlation heatmap (density/magnitude of data points within a dataset)
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm") # cool -> warm , annot shows the values of the squares
plt.show()
#Handle Missing Values

median_age = df["Age"].median()
df["Age"] = df["Age"].fillna(median_age)
df = df.dropna(subset=['Embarked'])

df.isnull().sum() #Checks the sum of no-value columns

#Encode Categorical Features

df.dtypes #Name & Ticket will be dropped. Name is unique, and Ticket has no predictive signal.

df["Sex"] = df["Sex"].map({"male" : 0, "female" : 1})

df_encoded = pd.get_dummies(df["Embarked"],prefix="Embarked",drop_first=True,dtype=int)

df = pd.concat([df_encoded,df],axis=1).drop(columns=["Embarked"])



#Produce X and Y
df = df.drop(columns=["Name","Ticket"])


y = df["Survived"]
X = df.drop(columns=["Survived"])

print(X.shape, X.dtypes)
