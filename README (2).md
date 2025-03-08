# Early-detection-of-chronic-kidney-disease

## Code

```py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv("kidney_disease.csv")

df.columns = ['id','age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
              'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']

df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
for col in cat_cols:
    df[col] = pd.Categorical(df[col]).codes
X = df.drop(columns='classification')

y = df['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 96.3:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
ckd_counts = df['classification'].value_counts()

plt.figure(figsize=(8, 6))
ckd_counts.plot(kind='bar', color=['blue', 'green'])
plt.title('Distribution of Patients: Damaged Cells vs Non-Damaged Cells')
plt.ylabel('Number of Patients')
plt.xticks([0, 1], ['Non-Damaged Cells (Healthy)', 'Damaged Cells (CKD)'], rotation=0)
plt.show()

```

## Output

![project 1](https://github.com/user-attachments/assets/056cea87-e3d3-4d39-907d-9243608b86a3)


![project 2](https://github.com/user-attachments/assets/96ad031b-bd89-415c-9bff-af04b5e40a9d)

## Result

We developed a machine learning model that have acheived 96.3 % accuracy using Decision Tree Classifier algorithm and also distribute the damaged and undamaged cells by visualzation graph using python.



