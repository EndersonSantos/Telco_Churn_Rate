from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from numpy import empty
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load

SVM_churn_model_prediction = load("telco_churn_predictor_model.joblib")

cat_attributes = ["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
num_attributes = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]
# Create a class to select numerical or categorical columns 

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

from sklearn.pipeline import Pipeline
num_pipeline = Pipeline([
        ('selector',DataFrameSelector(num_attributes)),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector',DataFrameSelector(cat_attributes)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

df = pd.read_csv("clean_churn.csv")
telco_prepared = full_pipeline.fit_transform(df)


app = Flask(__name__)
secret_key = 'e6187c406a23bcb4885b696471d619eb5ccb5e877acf75b3'
app.config['SECRET_KEY'] = secret_key

messages = [  ]

@app.route('/', methods=['GET'])
def create():
    return render_template('form.html') 

@app.route('/', methods=['POST'])
def created():
    gender = request.form['gender']
    partner = request.form['Partner']
    dependents = request.form['Dependents']
    Phone_Service = request.form['Phone Service']
    Multiple_Lines = request.form['Multiple Lines']
    Internet_Service = request.form['Internet Service']
    Online_Security = request.form['Online Security']
    Online_Backup = request.form['Online Backup']
    Device_Protection = request.form['Device Protection']
    Tech_Support = request.form['Tech Support']
    StreamingTV = request.form['StreamingTV']
    StreamingMovies = request.form['StreamingMovies']
    Paperless_Billing = request.form['Paperless Billing']
    Contract_Type = request.form['Contract Type']
    Payment_Method = request.form['Payment Method']
    tenure = request.form['tenure']
    Monthly_Charges = request.form['Monthly Charges']
    total_charges = int(tenure) * float(Monthly_Charges)
    Senior_Citizen = request.form['Senior Citizen']
    
    if Senior_Citizen == "Yes":
        citizen = 0
    else:
        citizen = 1
    
    tryed = {
    'gender':gender,
    'SeniorCitizen':[citizen],
    'Partner':partner,
    'Dependents':dependents,
    'tenure':[int(tenure)],
    'PhoneService':Phone_Service,
    'MultipleLines': Multiple_Lines,
    'InternetService':Internet_Service,
    'OnlineSecurity':Online_Security,
    'OnlineBackup':Online_Backup,
    'DeviceProtection':Device_Protection,
    'TechSupport':Tech_Support,
    'StreamingTV':StreamingTV,
    'StreamingMovies':StreamingMovies,
    'Contract':Contract_Type,
    'PaperlessBilling':Paperless_Billing,
    'PaymentMethod': Payment_Method,
    'MonthlyCharges':[float(Monthly_Charges)],
    'TotalCharges': [float(total_charges)]
    }
    
    tryed_test = pd.DataFrame(tryed)
    tryed_telco_prepared = full_pipeline.transform(tryed_test)
    prediction = SVM_churn_model_prediction.predict(tryed_telco_prepared)

    response = {"Churn": prediction[0]} 
    messages.append(response)

    return render_template("index.html", messages=messages)
        #return redirect(url_for('index'))


app.run()