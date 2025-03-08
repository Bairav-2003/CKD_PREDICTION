# CKD_PREDICTION


## Code

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

kidney=pd.read_csv('kidney_disease.csv')
kidney.shape
kidney.head()
kidney.info()
kidney.describe()

columns=pd.read_csv("data_description.txt",sep='-')
columns=columns.reset_index()
columns.columns=['cols','abb_col_names']
kidney.columns=columns['abb_col_names'].values

def convert_dtype(kidney,feature):
    kidney[feature]=pd.to_numeric(kidney[feature],errors='coerce')

features=['packed cell volume','white blood cell count','red blood cell count']
for i in features:
    convert_dtype(kidney,i)

kidney.drop('id',inplace=True,axis=1)

def extract_cat_num(kidney):
    cat_col=[col for col in kidney.columns if kidney[col].dtype=='O']
    num_col=[col for col in kidney.columns if kidney[col].dtype!='O']
    return cat_col,num_col

cat_col,num_col=extract_cat_num(kidney)

kidney['diabetes mellitus'].replace(to_replace={'\tno':'no','\tyes':'yes'},inplace=True)
kidney['coronary artery disease'].replace(to_replace={'\tno':'no'},inplace=True)
kidney['class'].replace(to_replace={'ckd\t':'ckd'},inplace=True)

plt.figure(figsize=(30,30))
for i,feature in enumerate(num_col):
    plt.subplot(5,3,i+1)
    kidney[feature].hist()
    plt.title(feature)

plt.figure(figsize=(20, 20))
for i, feature in enumerate(cat_col):
    plt.subplot(4, 3, i + 1)
    sns.countplot(x=kidney[feature])
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 20))
for i, feature in enumerate(cat_col):
    plt.subplot(4, 3, i + 1)
    sns.countplot(x=kidney[feature], hue=kidney['class'])
plt.tight_layout()
plt.show()

sns.countplot(x=kidney['class'])  
plt.show()

corr_matrix = kidney.corr(numeric_only=True)
plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, annot=True, cmap="BuPu", cbar=True, linewidths=0.5, fmt=".3f")

kidney.isnull().sum()
for i in num_col:
    kidney[i].fillna(kidney[i].median(),inplace=True)
kidney.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in cat_col:
    kidney[col]=le.fit_transform(kidney[col])

from sklearn.feature_selection import SelectKBest, chi2
ind_col=[col for col in kidney.columns if col!='class']
dep_col='class'
X=kidney[ind_col]
y=kidney[dep_col]

imp_features=SelectKBest(score_func=chi2,k=20)
imp_features=imp_features.fit(X,y)
features_rank=pd.DataFrame({'features': X.columns, 'score': imp_features.scores_})
selected=features_rank.nlargest(10,'score')['features'].values
X_new=kidney[selected]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_new,y,random_state=0,test_size=0.3)

from xgboost import XGBClassifier
params={'learning-rate':[0,0.5,0.20,0.25],'max_depth':[5,8,10],'min_child_weight':[1,3,5,7],'gamma':[0.0,0.1,0.2,0.4],'colsample_bytree':[0.3,0.4,0.7]}
from sklearn.model_selection import RandomizedSearchCV
classifier=XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
random_search.fit(X_train,y_train)
classifier=random_search.best_estimator_
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)


## Output

![project 1](https://github.com/user-attachments/assets/056cea87-e3d3-4d39-907d-9243608b86a3)



## Result

We developed a machine learning model that have acheived 98% accuracy using xboost algorithm and also distribute the damaged and undamaged cells by visualzation graph using python.



