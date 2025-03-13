# CKD_PREDICTION


## Code for ckd_prediction

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# For Filtering the warnings
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('kidney_disease.csv')

data.head()

data.info()

data.classification.unique()

data.classification=data.classification.replace("ckd\t","ckd") 

data.classification.unique()

data.drop('id', axis = 1, inplace = True)

data.head()

data['classification'] = data['classification'].replace(['ckd','notckd'], [1,0])

data.head()

data.isnull().sum()

df = data.dropna(axis = 0)
print(f"Before dropping all NaN values: {data.shape}")
print(f"After dropping all NaN values: {df.shape}")

df.head()

df.index = range(0,len(df),1)
df.head()

for i in df['wc']:
    print(i)

df['wc']=df['wc'].replace(["\t6200","\t8400"],[6200,8400])

for i in df['wc']:
    print(i)

df.info()

df['pcv']=df['pcv'].astype(int)
df['wc']=df['wc'].astype(int)
df['rc']=df['rc'].astype(float)
df.info()

object_dtypes = df.select_dtypes(include = 'object')
object_dtypes.head()

dictonary = {
        "rbc": {
        "abnormal":1,
        "normal": 0,
    },
        "pc":{
        "abnormal":1,
        "normal": 0,
    },
        "pcc":{
        "present":1,
        "notpresent":0,
    },
        "ba":{
        "notpresent":0,
        "present": 1,
    },
        "htn":{
        "yes":1,
        "no": 0,
    },
        "dm":{
        "yes":1,
        "no":0,
    },
        "cad":{
        "yes":1,
        "no": 0,
    },
        "appet":{
        "good":1,
        "poor": 0,
    },
        "pe":{
        "yes":1,
        "no":0,
    },
        "ane":{
        "yes":1,
        "no":0,
    }
}

df=df.replace(dictonary)

df.head()

import seaborn as sns
plt.figure(figsize = (20,20))
sns.heatmap(df.corr(), annot = True, fmt=".2f",linewidths=0.5)

df.corr()

X = df.drop(['classification', 'sg', 'appet', 'rc', 'pcv', 'hemo', 'sod'], axis = 1)
y = df['classification']

X.columns

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 20)
model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score

confusion_matrix(y_test, model.predict(X_test))

print(f"Accuracy is {round(accuracy_score(y_test, model.predict(X_test))*100, 2)}%")

input_data = []
for column in X.columns:
    value = input(f"Enter value for {column}: ")
    input_data.append(float(value))

input_array = np.array(input_data).reshape(1, -1)
prediction = model.predict(input_array)

if prediction[0] == 1:
    print("The person has kidney disease.")
else:
    print("The person does not have kidney disease.")

import os
import pickle
import joblib

# Save using pickle
with open('kidney_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save using joblib
joblib.dump(model, 'kidney_model_joblib.pkl')

print("Model saved successfully using both pickle and joblib!")

# Loading with pickle
with open('kidney_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Loading with joblib
loaded_model = joblib.load('kidney_model_joblib.pkl')
print(loaded_model)

```
## Code for classification
```py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
import cv2
from pathlib import Path
sns.set()

data_dir = Path(r'C:\Users\makes\Desktop\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone')
train_dir = data_dir

# Get the path to the normal and pneumonia sub-directories
Normal_Cases_dir = train_dir / 'Normal'
Cyst_Cases_dir = train_dir / 'Cyst'
Stone_Cases_dir = train_dir / 'Stone'
Tumor_Cases_dir = train_dir / 'Tumor'

# Getting the list of all the images
Normal_Cases = Normal_Cases_dir.glob('*.jpg')
Cyst_Cases = Cyst_Cases_dir.glob('*.jpg')
Stone_Cases = Stone_Cases_dir.glob('*.jpg')
Tumor_Cases = Tumor_Cases_dir.glob('*.jpg')

# An empty list for inserting data into this list in (image_path, Label) format
train_data = []


# Labeling the Cyst case as 0
for img in Cyst_Cases:
    train_data.append((img, 0))

# Labeling the Normal case as 1
for img in Normal_Cases:
    train_data.append((img, 1))

# Labeling the Stone case as 2
for img in Stone_Cases:
    train_data.append((img, 2))

# Labeling the Tumor case as 3
for img in Tumor_Cases:
    train_data.append((img, 3))

# Creating a DataFrame
train_df = pd.DataFrame(train_data, columns=['image', 'label'])

print(train_df.head())

# Visualizing the count of each class
plt.figure(figsize=(6, 4))
sns.countplot(x=train_df['label'])
plt.xlabel("Condition")
plt.ylabel("Number of Images")
plt.title("Distribution of Kidney Conditions in Dataset")
plt.show()

# Load and preprocess images
def preprocess_image(image_path):
    img = imread(str(image_path))
    img = cv2.resize(img, (150, 150))  # Resize images to 150x150
    img = img / 255.0  # Normalize pixel values
    return img

# Apply preprocessing
train_df['processed_image'] = train_df['image'].apply(preprocess_image)

# Splitting the dataset
from sklearn.model_selection import train_test_split

X = np.array(train_df['processed_image'].tolist())
y = np.array(train_df['label'].tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshaping for CNN input
X_train = X_train.reshape(-1, 150, 150, 3)
X_test = X_test.reshape(-1, 150, 150, 3)

# Building CNN Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluating the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Plot Training History
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# Save model
model.save('kidney_cnn_model.h5')
print("Model saved successfully!")

```

## Output
## for Randomforest
![project 1](https://github.com/Bairav-2003/CKD_PREDICTION/blob/main/xboostout1.png)


![project 2](https://github.com/Bairav-2003/CKD_PREDICTION/blob/main/xboostoutput2.png)
![project 3](https://github.com/Bairav-2003/CKD_PREDICTION/blob/main/randomoutput.png)

## for cnn
![project 2](https://github.com/Bairav-2003/CKD_PREDICTION/blob/main/cnn_output.png)

## webpage output
![project 2](https://github.com/Bairav-2003/CKD_PREDICTION/blob/main/h1.png)


![project 2](https://github.com/Bairav-2003/CKD_PREDICTION/blob/main/h2.png)

![project 2](https://github.com/Bairav-2003/CKD_PREDICTION/blob/main/Screenshot%202025-03-13%20155003.png)


![project 2](https://github.com/Bairav-2003/CKD_PREDICTION/blob/main/Screenshot%202025-03-13%20155023.png)

## Result

We developed a machine learning model that have acheived 98% accuracy using Random forest algorithm and also distribute the damaged and undamaged cells by visualzation graph using python and then we also classify the disease in the kindney using cnn.



