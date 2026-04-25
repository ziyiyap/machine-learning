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
plt.close()
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
test = X.join(y, how='right')
# Turn X and Y into vectors and matrices

sigmoid = lambda z: 1 / (1 + np.exp(-z))
BCEL = lambda y_pred, y_actual : -(np.sum(y_actual * np.log(y_pred) + (1-y_actual) * np.log(1 - y_pred))) / len(y_actual)

def add_ones(arr):
    ones = np.ones((arr.shape[0],1))
    return np.hstack((ones, arr))

def standardization(arr):
    mean_matrix = arr.mean(axis=0)
    stdv_matrix =arr.std(axis=0)
    return (arr - mean_matrix)/stdv_matrix, mean_matrix, stdv_matrix

def compute_gradients(X, y_pred, y_actual):
    dx = (X.T @ (y_pred - y_actual)) / X.shape[0]
    return dx

def train(X_, y, alpha = 0.1, epochs = 1000):
    w = np.zeros((X_.shape[1],1))

    for epoch in range(0, epochs+1):
        z = X_ @ w
        y_pred = sigmoid(z)
        loss = BCEL(y_pred, y)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} Entropy loss: {loss}")

        dw = compute_gradients(X_, y_pred, y)
        w -= alpha * dw
    return w

def predict(W, mean, stdv, embark_q = 0, embark_s = 0, pclass = 0, sex = 0, age = 0.0, sibsp = 0, parch = 0, fare = 0):

    features = pd.DataFrame([[1,embark_q,embark_s,pclass,sex,age,sibsp,parch,fare]], columns=["Bias", "Embarked_Q", "Embarked_S", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])
    features[["Age", "Fare", "SibSp", "Parch"]] = (features[["Age", "Fare", "SibSp", "Parch"]].to_numpy() - mean)/stdv

    z = features.to_numpy() @ W
    y_pred = sigmoid(z)
    threshold = 0.5
    
    if (y_pred >= threshold).astype(int).flatten().tolist()[0]:
        return "Survived"
    return "Skill issue"

X_copy = X.copy()
X_copy[["Age", "Fare", "SibSp", "Parch"]], meanv, stdvv = standardization(X_copy[["Age", "Fare", "SibSp", "Parch"]].to_numpy())

Xnp = X_copy.to_numpy()
Xv = add_ones(Xnp)
y_actual = y.to_numpy().reshape(-1,1)

W = train(Xv, y_actual)

print(predict(W,meanv, stdvv,0,1,3,0,22,1,0,7.25))