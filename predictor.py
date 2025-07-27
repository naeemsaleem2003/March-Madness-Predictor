import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv('data/tournament_data.csv')

# Basic EDA
print("Data Summary:")
print(df.describe())
print("\nNull values:")
print(df.isnull().sum())

# Feature engineering
df['seed_diff'] = df['team_seed'] - df['opp_seed']
df['point_diff'] = df['team_points'] - df['opp_points']

# Features and label
features = ['team_seed', 'opp_seed', 'team_points', 'opp_points', 'seed_diff', 'point_diff']
X = df[features]
y = df['win']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)

# Random Forest with hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20]
}
rf = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

# Evaluation
print(f"Logistic Regression Accuracy: {lr_acc:.2f}")
print(f"Random Forest Accuracy: {rf_acc:.2f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_preds))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, rf_preds)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
