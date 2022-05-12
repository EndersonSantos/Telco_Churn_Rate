```python
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import pandas as pd
pd.options.display.max_columns = 50
import seaborn as sns
# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "Project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR,"ScikitLearn-Book", "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```


```python
TELCO_PATH = os.path.join("datasets", "Telco Churn")
def load_telco_data(filename, telco_path=TELCO_PATH, extension="csv"):
    csv_path = os.path.join(telco_path, filename)
    return pd.read_csv(csv_path+"."+extension)
```


```python
df = load_telco_data("Telco-Customer-Churn")
```


```python
df = df.drop("customerID",axis=1)
```


```python
df["TotalCharges"].replace(" ", "0", inplace=True)
df["TotalCharges"] = df["TotalCharges"].apply(lambda x: float(x))
```


```python
X = df.drop("Churn",axis=1)
y = df["Churn"]
```


```python
X.to_csv("C:/Users/ewill/OneDrive/Desktop/telco_churn_flask_app/clean_churn.csv")
```


```python
from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
```


```python
X_train = train_set.drop("Churn",axis=1)
y_train = train_set["Churn"]
```


```python
X_test = test_set.drop("Churn",axis=1)
y_test = test_set["Churn"]
```


```python
cat_attributes = ["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
num_attributes = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]
```


```python
cat_atr = X[cat_attributes]
num_atr = X[num_attributes]
```


```python
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(sparse=False)
telco_cat_1hot = cat_encoder.fit_transform(cat_atr)
```


```python
telco_cat_1hot
```




    array([[1., 0., 0., ..., 0., 1., 0.],
           [0., 1., 1., ..., 0., 0., 1.],
           [0., 1., 1., ..., 0., 0., 1.],
           ...,
           [1., 0., 0., ..., 0., 1., 0.],
           [0., 1., 0., ..., 0., 0., 1.],
           [0., 1., 1., ..., 0., 0., 0.]])




```python
cat_encoder.categories_
```




    [array(['Female', 'Male'], dtype=object),
     array(['No', 'Yes'], dtype=object),
     array(['No', 'Yes'], dtype=object),
     array(['No', 'Yes'], dtype=object),
     array(['No', 'No phone service', 'Yes'], dtype=object),
     array(['DSL', 'Fiber optic', 'No'], dtype=object),
     array(['No', 'No internet service', 'Yes'], dtype=object),
     array(['No', 'No internet service', 'Yes'], dtype=object),
     array(['No', 'No internet service', 'Yes'], dtype=object),
     array(['No', 'No internet service', 'Yes'], dtype=object),
     array(['No', 'No internet service', 'Yes'], dtype=object),
     array(['No', 'No internet service', 'Yes'], dtype=object),
     array(['Month-to-month', 'One year', 'Two year'], dtype=object),
     array(['No', 'Yes'], dtype=object),
     array(['Bank transfer (automatic)', 'Credit card (automatic)',
            'Electronic check', 'Mailed check'], dtype=object)]




```python
from sklearn.preprocessing import StandardScaler
```


```python
from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
```


```python
from sklearn.pipeline import Pipeline
num_pipeline = Pipeline([
        ('selector',DataFrameSelector(num_attributes)),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector',DataFrameSelector(cat_attributes)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])
```


```python
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
```


```python
telco_prepared = full_pipeline.fit_transform(X_train)
```

# Logistic Regression


```python
from sklearn.linear_model import LogisticRegressionCV
```


```python
log_model = LogisticRegressionCV(solver='liblinear',cv=3)
log_model.fit(telco_prepared, y_train)
```




    LogisticRegressionCV(cv=3, solver='liblinear')




```python
predictions = log_model.predict(telco_prepared)
```


```python
from sklearn.metrics import confusion_matrix, precision_score, recall_score,precision_recall_curve,roc_curve,f1_score
```


```python
confusion_matrix(y_train,predictions)
```




    array([[3713,  425],
           [ 689,  807]], dtype=int64)




```python
recall_average = recall_score(y_train, predictions, average="binary", pos_label="Yes")
recall_average
```




    0.5394385026737968




```python
precision_score(y_train, predictions,pos_label="Yes")
```




    0.6550324675324676




```python
f1_score(y_train, predictions,pos_label="Yes")
```




    0.591642228739003



# Random Forest 


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
```


```python
rnd_clf.fit(telco_prepared,y_train)
```




    RandomForestClassifier(max_leaf_nodes=16, n_estimators=500, n_jobs=-1)




```python
prediction_rnd = rnd_clf.predict(telco_prepared)
```


```python
confusion_matrix(y_train,prediction_rnd)
```




    array([[3869,  269],
           [ 837,  659]], dtype=int64)




```python
precision_score(y_train, prediction_rnd, pos_label="Yes")
```




    0.7101293103448276




```python
recall_score(y_train, prediction_rnd, pos_label="Yes")
```




    0.4405080213903743




```python
f1_score(y_train, prediction_rnd,pos_label="Yes")
```




    0.5437293729372937




```python
features_importances = rnd_clf.feature_importances_
```


```python
for name , score in zip(X.columns,features_importances):
    print(name," : ", score)
```

    gender  :  0.0028193027185595275
    SeniorCitizen  :  0.1488187216756413
    Partner  :  0.03914094744264253
    Dependents  :  0.07960604163216391
    tenure  :  0.00023900135475003674
    PhoneService  :  0.00022956530270804625
    MultipleLines  :  0.0013440327104536946
    InternetService  :  0.0011364894303160236
    OnlineSecurity  :  0.002197527308129516
    OnlineBackup  :  0.00173683709964016
    DeviceProtection  :  0.0006177295435295412
    TechSupport  :  0.0007270367782919001
    StreamingTV  :  0.0018435702895642934
    StreamingMovies  :  0.0009824466026221686
    Contract  :  0.0026165561742787897
    PaperlessBilling  :  0.024664301916622697
    PaymentMethod  :  0.06337017945981056
    MonthlyCharges  :  0.011236019069966456
    TotalCharges  :  0.08948733723127116
    

# Support Vector Machines


```python
from sklearn.svm import SVC
```


```python
svm_clf = SVC(kernel="poly", degree=3, coef0=1, C=5)
```


```python
svm_clf.fit(telco_prepared,y_train)
```




    SVC(C=5, coef0=1, kernel='poly')




```python
predictions_svm = svm_clf.predict(telco_prepared)
```


```python
confusion_matrix(y_train,prediction_rnd)
```




    array([[3869,  269],
           [ 837,  659]], dtype=int64)




```python
recall_score(y_train, predictions_svm, pos_label="Yes")
```




    0.6336898395721925




```python
precision_score(y_train, predictions_svm, pos_label="Yes")
```




    0.7423649177760376




```python
f1_score(y_train, predictions_svm,pos_label="Yes")
```




    0.6837360259646592




```python
svm_clf.decision_function(telco_prepared)
```




    array([-2.39300359, -0.65504135,  1.1055693 , ..., -1.02639803,
            1.9639709 , -1.193322  ])



## SVM with grid search


```python
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf'],
             }
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
```


```python
grid.fit(telco_prepared, y_train)
```

    Fitting 5 folds for each of 25 candidates, totalling 125 fits
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    

    [CV] .......... C=0.1, gamma=1, kernel=rbf, score=0.736, total=   3.7s
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    3.6s remaining:    0.0s
    

    [CV] .......... C=0.1, gamma=1, kernel=rbf, score=0.736, total=   3.7s
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    

    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    7.2s remaining:    0.0s
    

    [CV] .......... C=0.1, gamma=1, kernel=rbf, score=0.736, total=   3.6s
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    [CV] .......... C=0.1, gamma=1, kernel=rbf, score=0.736, total=   3.4s
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    [CV] .......... C=0.1, gamma=1, kernel=rbf, score=0.734, total=   3.7s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=0.1, gamma=0.1, kernel=rbf, score=0.797, total=   2.3s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=0.1, gamma=0.1, kernel=rbf, score=0.802, total=   2.4s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=0.1, gamma=0.1, kernel=rbf, score=0.784, total=   2.2s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=0.1, gamma=0.1, kernel=rbf, score=0.789, total=   2.1s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=0.1, gamma=0.1, kernel=rbf, score=0.798, total=   2.1s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=0.1, gamma=0.01, kernel=rbf, score=0.791, total=   2.2s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=0.1, gamma=0.01, kernel=rbf, score=0.797, total=   2.3s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=0.1, gamma=0.01, kernel=rbf, score=0.788, total=   2.2s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=0.1, gamma=0.01, kernel=rbf, score=0.787, total=   2.5s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=0.1, gamma=0.01, kernel=rbf, score=0.782, total=   2.4s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=0.1, gamma=0.001, kernel=rbf, score=0.735, total=   2.2s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=0.1, gamma=0.001, kernel=rbf, score=0.735, total=   2.1s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=0.1, gamma=0.001, kernel=rbf, score=0.735, total=   2.1s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=0.1, gamma=0.001, kernel=rbf, score=0.734, total=   2.1s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=0.1, gamma=0.001, kernel=rbf, score=0.734, total=   2.2s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=0.1, gamma=0.0001, kernel=rbf, score=0.735, total=   2.3s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=0.1, gamma=0.0001, kernel=rbf, score=0.735, total=   2.3s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=0.1, gamma=0.0001, kernel=rbf, score=0.735, total=   2.1s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=0.1, gamma=0.0001, kernel=rbf, score=0.734, total=   2.0s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=0.1, gamma=0.0001, kernel=rbf, score=0.734, total=   1.9s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV] ............ C=1, gamma=1, kernel=rbf, score=0.774, total=   4.1s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV] ............ C=1, gamma=1, kernel=rbf, score=0.762, total=   4.2s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV] ............ C=1, gamma=1, kernel=rbf, score=0.757, total=   4.1s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV] ............ C=1, gamma=1, kernel=rbf, score=0.755, total=   3.9s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV] ............ C=1, gamma=1, kernel=rbf, score=0.757, total=   4.2s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV] .......... C=1, gamma=0.1, kernel=rbf, score=0.807, total=   2.3s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV] .......... C=1, gamma=0.1, kernel=rbf, score=0.801, total=   2.2s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV] .......... C=1, gamma=0.1, kernel=rbf, score=0.791, total=   2.3s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV] .......... C=1, gamma=0.1, kernel=rbf, score=0.788, total=   2.2s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV] .......... C=1, gamma=0.1, kernel=rbf, score=0.794, total=   2.3s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV] ......... C=1, gamma=0.01, kernel=rbf, score=0.800, total=   2.0s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV] ......... C=1, gamma=0.01, kernel=rbf, score=0.813, total=   2.0s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV] ......... C=1, gamma=0.01, kernel=rbf, score=0.787, total=   1.9s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV] ......... C=1, gamma=0.01, kernel=rbf, score=0.795, total=   2.1s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV] ......... C=1, gamma=0.01, kernel=rbf, score=0.798, total=   2.4s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV] ........ C=1, gamma=0.001, kernel=rbf, score=0.791, total=   2.4s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV] ........ C=1, gamma=0.001, kernel=rbf, score=0.801, total=   2.0s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV] ........ C=1, gamma=0.001, kernel=rbf, score=0.785, total=   2.1s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV] ........ C=1, gamma=0.001, kernel=rbf, score=0.791, total=   2.2s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV] ........ C=1, gamma=0.001, kernel=rbf, score=0.790, total=   2.3s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV] ....... C=1, gamma=0.0001, kernel=rbf, score=0.735, total=   2.4s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV] ....... C=1, gamma=0.0001, kernel=rbf, score=0.735, total=   2.5s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV] ....... C=1, gamma=0.0001, kernel=rbf, score=0.735, total=   2.5s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV] ....... C=1, gamma=0.0001, kernel=rbf, score=0.734, total=   2.2s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV] ....... C=1, gamma=0.0001, kernel=rbf, score=0.734, total=   2.5s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV] ........... C=10, gamma=1, kernel=rbf, score=0.774, total=   5.9s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV] ........... C=10, gamma=1, kernel=rbf, score=0.757, total=   6.1s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV] ........... C=10, gamma=1, kernel=rbf, score=0.757, total=   5.3s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV] ........... C=10, gamma=1, kernel=rbf, score=0.761, total=   5.4s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV] ........... C=10, gamma=1, kernel=rbf, score=0.757, total=   5.7s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV] ......... C=10, gamma=0.1, kernel=rbf, score=0.773, total=   3.1s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV] ......... C=10, gamma=0.1, kernel=rbf, score=0.765, total=   3.3s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV] ......... C=10, gamma=0.1, kernel=rbf, score=0.759, total=   3.5s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV] ......... C=10, gamma=0.1, kernel=rbf, score=0.760, total=   3.5s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV] ......... C=10, gamma=0.1, kernel=rbf, score=0.753, total=   4.0s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV] ........ C=10, gamma=0.01, kernel=rbf, score=0.806, total=   2.7s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV] ........ C=10, gamma=0.01, kernel=rbf, score=0.806, total=   2.6s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV] ........ C=10, gamma=0.01, kernel=rbf, score=0.782, total=   2.9s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV] ........ C=10, gamma=0.01, kernel=rbf, score=0.794, total=   2.7s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV] ........ C=10, gamma=0.01, kernel=rbf, score=0.801, total=   2.8s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV] ....... C=10, gamma=0.001, kernel=rbf, score=0.799, total=   2.4s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV] ....... C=10, gamma=0.001, kernel=rbf, score=0.810, total=   2.4s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV] ....... C=10, gamma=0.001, kernel=rbf, score=0.791, total=   2.3s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV] ....... C=10, gamma=0.001, kernel=rbf, score=0.791, total=   2.7s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV] ....... C=10, gamma=0.001, kernel=rbf, score=0.801, total=   2.7s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV] ...... C=10, gamma=0.0001, kernel=rbf, score=0.791, total=   2.6s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV] ...... C=10, gamma=0.0001, kernel=rbf, score=0.803, total=   2.5s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV] ...... C=10, gamma=0.0001, kernel=rbf, score=0.784, total=   2.5s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV] ...... C=10, gamma=0.0001, kernel=rbf, score=0.791, total=   2.6s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV] ...... C=10, gamma=0.0001, kernel=rbf, score=0.794, total=   2.5s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV] .......... C=100, gamma=1, kernel=rbf, score=0.771, total=   5.8s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV] .......... C=100, gamma=1, kernel=rbf, score=0.756, total=   4.9s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV] .......... C=100, gamma=1, kernel=rbf, score=0.760, total=   6.7s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV] .......... C=100, gamma=1, kernel=rbf, score=0.748, total=   5.8s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV] .......... C=100, gamma=1, kernel=rbf, score=0.756, total=   5.2s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=100, gamma=0.1, kernel=rbf, score=0.742, total=   5.9s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=100, gamma=0.1, kernel=rbf, score=0.729, total=   5.4s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=100, gamma=0.1, kernel=rbf, score=0.734, total=   5.1s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=100, gamma=0.1, kernel=rbf, score=0.731, total=   5.4s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=100, gamma=0.1, kernel=rbf, score=0.738, total=   5.4s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=100, gamma=0.01, kernel=rbf, score=0.804, total=   3.7s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=100, gamma=0.01, kernel=rbf, score=0.803, total=   3.7s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=100, gamma=0.01, kernel=rbf, score=0.787, total=   3.7s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=100, gamma=0.01, kernel=rbf, score=0.794, total=   3.8s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=100, gamma=0.01, kernel=rbf, score=0.802, total=   3.6s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=100, gamma=0.001, kernel=rbf, score=0.802, total=   2.7s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=100, gamma=0.001, kernel=rbf, score=0.811, total=   2.5s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=100, gamma=0.001, kernel=rbf, score=0.787, total=   2.4s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=100, gamma=0.001, kernel=rbf, score=0.798, total=   2.5s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=100, gamma=0.001, kernel=rbf, score=0.795, total=   2.5s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=100, gamma=0.0001, kernel=rbf, score=0.797, total=   2.4s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=100, gamma=0.0001, kernel=rbf, score=0.807, total=   2.3s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=100, gamma=0.0001, kernel=rbf, score=0.790, total=   2.3s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=100, gamma=0.0001, kernel=rbf, score=0.791, total=   2.5s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=100, gamma=0.0001, kernel=rbf, score=0.804, total=   2.4s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV] ......... C=1000, gamma=1, kernel=rbf, score=0.762, total=   9.2s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV] ......... C=1000, gamma=1, kernel=rbf, score=0.749, total=   6.9s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV] ......... C=1000, gamma=1, kernel=rbf, score=0.754, total=   6.6s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV] ......... C=1000, gamma=1, kernel=rbf, score=0.740, total=   9.3s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV] ......... C=1000, gamma=1, kernel=rbf, score=0.746, total=   8.5s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV] ....... C=1000, gamma=0.1, kernel=rbf, score=0.712, total=   9.7s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV] ....... C=1000, gamma=0.1, kernel=rbf, score=0.725, total=   8.5s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV] ....... C=1000, gamma=0.1, kernel=rbf, score=0.719, total=   8.5s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV] ....... C=1000, gamma=0.1, kernel=rbf, score=0.703, total=   7.3s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV] ....... C=1000, gamma=0.1, kernel=rbf, score=0.734, total=   7.5s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV] ...... C=1000, gamma=0.01, kernel=rbf, score=0.789, total=  10.8s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV] ...... C=1000, gamma=0.01, kernel=rbf, score=0.776, total=  11.3s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV] ...... C=1000, gamma=0.01, kernel=rbf, score=0.783, total=  10.5s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV] ...... C=1000, gamma=0.01, kernel=rbf, score=0.775, total=  10.6s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV] ...... C=1000, gamma=0.01, kernel=rbf, score=0.775, total=  11.2s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV] ..... C=1000, gamma=0.001, kernel=rbf, score=0.799, total=   4.0s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV] ..... C=1000, gamma=0.001, kernel=rbf, score=0.806, total=   3.6s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV] ..... C=1000, gamma=0.001, kernel=rbf, score=0.783, total=   3.8s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV] ..... C=1000, gamma=0.001, kernel=rbf, score=0.795, total=   3.4s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV] ..... C=1000, gamma=0.001, kernel=rbf, score=0.805, total=   3.8s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV] .... C=1000, gamma=0.0001, kernel=rbf, score=0.797, total=   2.8s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV] .... C=1000, gamma=0.0001, kernel=rbf, score=0.805, total=   2.6s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV] .... C=1000, gamma=0.0001, kernel=rbf, score=0.791, total=   2.6s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV] .... C=1000, gamma=0.0001, kernel=rbf, score=0.791, total=   2.5s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV] .... C=1000, gamma=0.0001, kernel=rbf, score=0.802, total=   2.2s
    

    [Parallel(n_jobs=1)]: Done 125 out of 125 | elapsed:  8.0min finished
    




    GridSearchCV(estimator=SVC(),
                 param_grid={'C': [0.1, 1, 10, 100, 1000],
                             'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                             'kernel': ['rbf']},
                 verbose=3)




```python
grid.best_params_
```




    {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}




```python
predictions = grid.predict(telco_prepared)
```


```python
confusion_matrix(y_train,prediction_rnd)
```




    array([[3869,  269],
           [ 837,  659]], dtype=int64)




```python
recall_score(y_train, predictions_svm, pos_label="Yes")
```




    0.6336898395721925




```python
precision_score(y_train, predictions_svm, pos_label="Yes")
```




    0.7423649177760376




```python
f1_score(y_train, predictions_svm,pos_label="Yes")
```




    0.6837360259646592




```python
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'degree' :[2,3,4],
              'kernel': ['poly']
             }
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
```


```python
grid.fit(telco_prepared, y_train)
```

    Fitting 5 folds for each of 15 candidates, totalling 75 fits
    [CV] C=0.1, degree=2, kernel=poly ....................................
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    

    [CV] ........ C=0.1, degree=2, kernel=poly, score=0.803, total=   0.7s
    [CV] C=0.1, degree=2, kernel=poly ....................................
    

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.6s remaining:    0.0s
    

    [CV] ........ C=0.1, degree=2, kernel=poly, score=0.809, total=   0.7s
    [CV] C=0.1, degree=2, kernel=poly ....................................
    

    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    1.4s remaining:    0.0s
    

    [CV] ........ C=0.1, degree=2, kernel=poly, score=0.784, total=   0.6s
    [CV] C=0.1, degree=2, kernel=poly ....................................
    [CV] ........ C=0.1, degree=2, kernel=poly, score=0.791, total=   0.6s
    [CV] C=0.1, degree=2, kernel=poly ....................................
    [CV] ........ C=0.1, degree=2, kernel=poly, score=0.800, total=   0.6s
    [CV] C=0.1, degree=3, kernel=poly ....................................
    [CV] ........ C=0.1, degree=3, kernel=poly, score=0.799, total=   0.7s
    [CV] C=0.1, degree=3, kernel=poly ....................................
    [CV] ........ C=0.1, degree=3, kernel=poly, score=0.804, total=   0.6s
    [CV] C=0.1, degree=3, kernel=poly ....................................
    [CV] ........ C=0.1, degree=3, kernel=poly, score=0.782, total=   0.6s
    [CV] C=0.1, degree=3, kernel=poly ....................................
    [CV] ........ C=0.1, degree=3, kernel=poly, score=0.792, total=   0.6s
    [CV] C=0.1, degree=3, kernel=poly ....................................
    [CV] ........ C=0.1, degree=3, kernel=poly, score=0.798, total=   0.7s
    [CV] C=0.1, degree=4, kernel=poly ....................................
    [CV] ........ C=0.1, degree=4, kernel=poly, score=0.801, total=   0.7s
    [CV] C=0.1, degree=4, kernel=poly ....................................
    [CV] ........ C=0.1, degree=4, kernel=poly, score=0.799, total=   0.7s
    [CV] C=0.1, degree=4, kernel=poly ....................................
    [CV] ........ C=0.1, degree=4, kernel=poly, score=0.782, total=   0.7s
    [CV] C=0.1, degree=4, kernel=poly ....................................
    [CV] ........ C=0.1, degree=4, kernel=poly, score=0.795, total=   0.7s
    [CV] C=0.1, degree=4, kernel=poly ....................................
    [CV] ........ C=0.1, degree=4, kernel=poly, score=0.798, total=   0.7s
    [CV] C=1, degree=2, kernel=poly ......................................
    [CV] .......... C=1, degree=2, kernel=poly, score=0.801, total=   0.7s
    [CV] C=1, degree=2, kernel=poly ......................................
    [CV] .......... C=1, degree=2, kernel=poly, score=0.805, total=   0.7s
    [CV] C=1, degree=2, kernel=poly ......................................
    [CV] .......... C=1, degree=2, kernel=poly, score=0.785, total=   0.7s
    [CV] C=1, degree=2, kernel=poly ......................................
    [CV] .......... C=1, degree=2, kernel=poly, score=0.792, total=   0.6s
    [CV] C=1, degree=2, kernel=poly ......................................
    [CV] .......... C=1, degree=2, kernel=poly, score=0.801, total=   0.7s
    [CV] C=1, degree=3, kernel=poly ......................................
    [CV] .......... C=1, degree=3, kernel=poly, score=0.807, total=   0.8s
    [CV] C=1, degree=3, kernel=poly ......................................
    [CV] .......... C=1, degree=3, kernel=poly, score=0.795, total=   0.8s
    [CV] C=1, degree=3, kernel=poly ......................................
    [CV] .......... C=1, degree=3, kernel=poly, score=0.789, total=   0.8s
    [CV] C=1, degree=3, kernel=poly ......................................
    [CV] .......... C=1, degree=3, kernel=poly, score=0.792, total=   0.7s
    [CV] C=1, degree=3, kernel=poly ......................................
    [CV] .......... C=1, degree=3, kernel=poly, score=0.793, total=   0.8s
    [CV] C=1, degree=4, kernel=poly ......................................
    [CV] .......... C=1, degree=4, kernel=poly, score=0.807, total=   0.9s
    [CV] C=1, degree=4, kernel=poly ......................................
    [CV] .......... C=1, degree=4, kernel=poly, score=0.786, total=   0.8s
    [CV] C=1, degree=4, kernel=poly ......................................
    [CV] .......... C=1, degree=4, kernel=poly, score=0.785, total=   0.8s
    [CV] C=1, degree=4, kernel=poly ......................................
    [CV] .......... C=1, degree=4, kernel=poly, score=0.772, total=   0.8s
    [CV] C=1, degree=4, kernel=poly ......................................
    [CV] .......... C=1, degree=4, kernel=poly, score=0.782, total=   0.8s
    [CV] C=10, degree=2, kernel=poly .....................................
    [CV] ......... C=10, degree=2, kernel=poly, score=0.803, total=   1.2s
    [CV] C=10, degree=2, kernel=poly .....................................
    [CV] ......... C=10, degree=2, kernel=poly, score=0.801, total=   1.1s
    [CV] C=10, degree=2, kernel=poly .....................................
    [CV] ......... C=10, degree=2, kernel=poly, score=0.781, total=   1.1s
    [CV] C=10, degree=2, kernel=poly .....................................
    [CV] ......... C=10, degree=2, kernel=poly, score=0.786, total=   1.1s
    [CV] C=10, degree=2, kernel=poly .....................................
    [CV] ......... C=10, degree=2, kernel=poly, score=0.806, total=   1.1s
    [CV] C=10, degree=3, kernel=poly .....................................
    [CV] ......... C=10, degree=3, kernel=poly, score=0.791, total=   1.2s
    [CV] C=10, degree=3, kernel=poly .....................................
    [CV] ......... C=10, degree=3, kernel=poly, score=0.764, total=   1.2s
    [CV] C=10, degree=3, kernel=poly .....................................
    [CV] ......... C=10, degree=3, kernel=poly, score=0.769, total=   1.2s
    [CV] C=10, degree=3, kernel=poly .....................................
    [CV] ......... C=10, degree=3, kernel=poly, score=0.754, total=   1.2s
    [CV] C=10, degree=3, kernel=poly .....................................
    [CV] ......... C=10, degree=3, kernel=poly, score=0.771, total=   1.3s
    [CV] C=10, degree=4, kernel=poly .....................................
    [CV] ......... C=10, degree=4, kernel=poly, score=0.752, total=   1.5s
    [CV] C=10, degree=4, kernel=poly .....................................
    [CV] ......... C=10, degree=4, kernel=poly, score=0.744, total=   1.4s
    [CV] C=10, degree=4, kernel=poly .....................................
    [CV] ......... C=10, degree=4, kernel=poly, score=0.735, total=   1.4s
    [CV] C=10, degree=4, kernel=poly .....................................
    [CV] ......... C=10, degree=4, kernel=poly, score=0.744, total=   1.4s
    [CV] C=10, degree=4, kernel=poly .....................................
    [CV] ......... C=10, degree=4, kernel=poly, score=0.755, total=   1.4s
    [CV] C=100, degree=2, kernel=poly ....................................
    [CV] ........ C=100, degree=2, kernel=poly, score=0.800, total=   4.3s
    [CV] C=100, degree=2, kernel=poly ....................................
    [CV] ........ C=100, degree=2, kernel=poly, score=0.797, total=   4.6s
    [CV] C=100, degree=2, kernel=poly ....................................
    [CV] ........ C=100, degree=2, kernel=poly, score=0.782, total=   4.7s
    [CV] C=100, degree=2, kernel=poly ....................................
    [CV] ........ C=100, degree=2, kernel=poly, score=0.789, total=   4.2s
    [CV] C=100, degree=2, kernel=poly ....................................
    [CV] ........ C=100, degree=2, kernel=poly, score=0.806, total=   5.2s
    [CV] C=100, degree=3, kernel=poly ....................................
    [CV] ........ C=100, degree=3, kernel=poly, score=0.748, total=   6.0s
    [CV] C=100, degree=3, kernel=poly ....................................
    [CV] ........ C=100, degree=3, kernel=poly, score=0.733, total=   5.0s
    [CV] C=100, degree=3, kernel=poly ....................................
    [CV] ........ C=100, degree=3, kernel=poly, score=0.727, total=   5.7s
    [CV] C=100, degree=3, kernel=poly ....................................
    [CV] ........ C=100, degree=3, kernel=poly, score=0.722, total=   7.9s
    [CV] C=100, degree=3, kernel=poly ....................................
    [CV] ........ C=100, degree=3, kernel=poly, score=0.747, total=   6.6s
    [CV] C=100, degree=4, kernel=poly ....................................
    [CV] ........ C=100, degree=4, kernel=poly, score=0.715, total=   5.4s
    [CV] C=100, degree=4, kernel=poly ....................................
    [CV] ........ C=100, degree=4, kernel=poly, score=0.727, total=   3.2s
    [CV] C=100, degree=4, kernel=poly ....................................
    [CV] ........ C=100, degree=4, kernel=poly, score=0.722, total=   4.8s
    [CV] C=100, degree=4, kernel=poly ....................................
    [CV] ........ C=100, degree=4, kernel=poly, score=0.713, total=   3.2s
    [CV] C=100, degree=4, kernel=poly ....................................
    [CV] ........ C=100, degree=4, kernel=poly, score=0.729, total=   4.1s
    [CV] C=1000, degree=2, kernel=poly ...................................
    [CV] ....... C=1000, degree=2, kernel=poly, score=0.798, total=  48.3s
    [CV] C=1000, degree=2, kernel=poly ...................................
    [CV] ....... C=1000, degree=2, kernel=poly, score=0.797, total=  46.3s
    [CV] C=1000, degree=2, kernel=poly ...................................
    [CV] ....... C=1000, degree=2, kernel=poly, score=0.783, total=  44.2s
    [CV] C=1000, degree=2, kernel=poly ...................................
    [CV] ....... C=1000, degree=2, kernel=poly, score=0.785, total=  35.4s
    [CV] C=1000, degree=2, kernel=poly ...................................
    [CV] ....... C=1000, degree=2, kernel=poly, score=0.801, total=  51.8s
    [CV] C=1000, degree=3, kernel=poly ...................................
    [CV] ....... C=1000, degree=3, kernel=poly, score=0.714, total= 1.1min
    [CV] C=1000, degree=3, kernel=poly ...................................
    [CV] ....... C=1000, degree=3, kernel=poly, score=0.698, total=  58.1s
    [CV] C=1000, degree=3, kernel=poly ...................................
    [CV] ....... C=1000, degree=3, kernel=poly, score=0.716, total=  50.7s
    [CV] C=1000, degree=3, kernel=poly ...................................
    [CV] ....... C=1000, degree=3, kernel=poly, score=0.713, total=  59.6s
    [CV] C=1000, degree=3, kernel=poly ...................................
    [CV] ....... C=1000, degree=3, kernel=poly, score=0.728, total=  47.9s
    [CV] C=1000, degree=4, kernel=poly ...................................
    [CV] ....... C=1000, degree=4, kernel=poly, score=0.713, total=   7.7s
    [CV] C=1000, degree=4, kernel=poly ...................................
    [CV] ....... C=1000, degree=4, kernel=poly, score=0.713, total=   7.4s
    [CV] C=1000, degree=4, kernel=poly ...................................
    [CV] ....... C=1000, degree=4, kernel=poly, score=0.712, total=   9.4s
    [CV] C=1000, degree=4, kernel=poly ...................................
    [CV] ....... C=1000, degree=4, kernel=poly, score=0.703, total=   9.2s
    [CV] C=1000, degree=4, kernel=poly ...................................
    [CV] ....... C=1000, degree=4, kernel=poly, score=0.719, total=   6.3s
    

    [Parallel(n_jobs=1)]: Done  75 out of  75 | elapsed: 11.0min finished
    




    GridSearchCV(estimator=SVC(),
                 param_grid={'C': [0.1, 1, 10, 100, 1000], 'degree': [2, 3, 4],
                             'kernel': ['poly']},
                 verbose=3)




```python
predictions = grid.predict(telco_prepared)
```


```python
grid.best_params_
```




    {'C': 0.1, 'degree': 2, 'kernel': 'poly'}




```python
confusion_matrix(y_train,prediction_rnd)
```




    array([[3869,  269],
           [ 837,  659]], dtype=int64)




```python
recall_score(y_train, predictions_svm, pos_label="Yes")
```




    0.6336898395721925




```python
precision_score(y_train, predictions_svm, pos_label="Yes")
```




    0.7423649177760376




```python
f1_score(y_train, predictions_svm,pos_label="Yes")
```




    0.6837360259646592



## Testing the best model  on the test set


```python
telco_prepared_test = full_pipeline.fit_transform(X_test)
```


```python
test_pred = grid.predict(telco_prepared_test)
```


```python
confusion_matrix(y_test,test_pred)
```




    array([[960,  76],
           [183, 190]], dtype=int64)




```python
recall_score(y_test, test_pred, pos_label="Yes")
```




    0.5093833780160858




```python
precision_score(y_test, test_pred, pos_label="Yes")
```




    0.7142857142857143




```python
f1_score(y_test, test_pred,pos_label="Yes")
```




    0.594679186228482



## Retraining on the full dataset


```python
telco_prepared_full = full_pipeline.fit_transform(X)
```


```python
telco_prepared_full.shape
```




    (7043, 45)




```python
grid.best_params_ #'C': 1, 'gamma': 0.01, 'kernel': 'rbf'
```




    {'C': 0.1, 'degree': 2, 'kernel': 'poly'}




```python
svc_model = SVC(C=0.1,degree=2,kernel="poly")
```


```python
svc_model.fit(telco_prepared_full,y)
```




    SVC(C=0.1, degree=2, kernel='poly')




```python
tryed = {
    'gender':"Female",
    'SeniorCitizen':[0],
    'Partner':"Yes",
    'Dependents':"Yes",
    'tenure':3,
    'PhoneService':"No",
    'MultipleLines': "No phone service",
    'InternetService':"DSL",
    'OnlineSecurity':"Yes",
    'OnlineBackup':"No",
    'DeviceProtection':"Yes",
    'TechSupport':"Yes",
    'StreamingTV':"Yes",
    'StreamingMovies':"No",
    'Contract':"Month-to-month",
    'PaperlessBilling':"Yes",
    'PaymentMethod': "Electronic check",
    'MonthlyCharges':[53.85],
    'TotalCharges': [161.55]
}
```


```python
    tryed_test = pd.DataFrame(tryed)
```


```python
tryed_telco_prepared = full_pipeline.transform(tryed_test) #Only call the transform method
```


```python
tryed_telco_prepared.shape
```




    (1, 45)




```python
telco_prepared.shape
```




    (5634, 45)




```python
tryed_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>3</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>53.85</td>
      <td>161.55</td>
    </tr>
  </tbody>
</table>
</div>




```python
svc_model.predict(tryed_telco_prepared)
```




    array(['No'], dtype=object)




```python
from joblib import dump, load
```


```python
dump(svc_model, "telco_churn_predictor_model.joblib")
```




    ['telco_churn_predictor_model.joblib']




```python
loaded_model = load("telco_churn_predictor_model.joblib")
```


```python

```
