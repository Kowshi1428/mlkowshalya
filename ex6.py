import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("purchase.csv")

# Features and target
x = data.drop("Purchase", axis=1)
y = data["Purchase"]

# Split dataset
xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize and train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(xtr, ytr)

# Predictions
ypr = model.predict(xte)

# Evaluation
print("Accuracy:", accuracy_score(yte, ypr))
print("\nConfusion Matrix:\n", confusion_matrix(yte, ypr))
print("\nClassification Report:\n", classification_report(yte, ypr))

# Predict for new data point
new = np.array([[45, 48000, 17, 9]])  # Make sure the features order matches your dataset
pred = model.predict(new)
print("Will the customer purchase?", pred[0])

# Visualization
sns.set(style="whitegrid")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data=data, x="BrowsingTime", hue="Purchase", bins=6, kde=True, palette="Set1")
plt.title("Browsing Time Distribution by Purchase")

plt.subplot(1, 2, 2)
sns.histplot(data=data, x="Income", hue="Purchase", bins=6, kde=True, palette="Set2")
plt.title("Income Distribution by Purchase")

plt.tight_layout()
plt.show()
