import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
from pathlib import Path
import os
BASE_DIR = Path(__file__).parent
os.chdir(BASE_DIR)
# Load CSV
df = pd.read_csv("transaction.csv")

# Features
X_text = df["desc"]
X_price = df[["amount"]]

# Labels
y = df["Category"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer(lowercase=True)

X_text_vec = vectorizer.fit_transform(X_text)

# Combine text features + numeric price
X = hstack([X_text_vec, X_price.values])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Multinomial Logistic Regression
model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print(X_text_vec)

# New transaction
new_description = ["The Brew House"]
new_price = [[9.9]]

# Convert text using EXISTING vectorizer
new_text_vec = vectorizer.transform(new_description)

# Combine text + price
new_X = hstack([new_text_vec, new_price])

# Predict
pred = model.predict(new_X)

# Convert encoded label back to text
pred_label = label_encoder.inverse_transform(pred)

print(pred_label[0])