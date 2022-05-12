## We tryied to do some feature engineering and create new features to see if the performance of our model would improve but it didn't


```python
# Changing some features

def tenure(tenure_):
    if tenure_ <12:
        return "less than 1 year"
    elif tenure_>=12 and tenure_<24:
        return "1 to 2 years"
    elif tenure_>=24 and tenure_ <48:
            return "2 to 4 years"
    else:
        return "more than 4 year"

df["tenure_cat"] = df["tenure"].apply(tenure)

def total_services(phone, security, backup, projection, support, tv_stream, movie_stream):
    count = 1
    if phone == "Yes" :
        count += 1
        if security == "Yes":
            count += 1
            if backup == "Yes":
                count +=1
                if projection=="Yes":
                    count+=1
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                              
                else:
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
            else:
                if projection=="Yes":
                    count+=1
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                else:
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
        else:
            if backup == "Yes":
                count +=1
                if projection=="Yes":
                    count+=1
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                else:
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
            else:
                if projection=="Yes":
                    count+=1
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                else:
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
    else:
        if security == "Yes":
            count += 1
            if backup == "Yes":
                count +=1
                if projection=="Yes":
                    count+=1
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                else:
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
            else:
                if projection=="Yes":
                    count+=1
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                else:
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
        else:
            if backup == "Yes":
                count +=1
                if projection=="Yes":
                    count+=1
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                else:
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
            else:
                if projection=="Yes":
                    count+=1
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                else:
                    if support == "Yes":
                        count+=1
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
                    else:
                        if tv_stream == "Yes":
                            count+=1
                            if movie_stream=="Yes":
                                  count +=1
                        else:
                            if movie_stream=="Yes":
                                  count +=1
    return f"{count}"

df["Number_Services"] = df[["PhoneService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV", "StreamingMovies"]].apply(lambda df: total_services(df["PhoneService"],df["OnlineSecurity"],df["OnlineBackup"],df["DeviceProtection"],df["TechSupport"], df["StreamingTV"], df["StreamingMovies"]), axis=1)

df

df = df.drop(["PhoneService","OnlineSecurity","OnlineBackup",'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'],axis=1)

df

train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

X_train = train_set.drop("Churn",axis=1)
y_train = train_set["Churn"]
X_test = test_set.drop("Churn",axis=1)
y_test = test_set["Churn"]

cat_attributes = ["gender","Partner","Dependents","MultipleLines","Contract","PaperlessBilling","PaymentMethod","tenure_cat","Number_Services"]
num_attributes = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]

num_pipeline = Pipeline([
        ('selector',DataFrameSelector(num_attributes)),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector',DataFrameSelector(cat_attributes)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

telco_prepared = full_pipeline.fit_transform(X_train)

log_model = LogisticRegressionCV(solver='liblinear',cv=3)
log_model.fit(telco_prepared, y_train)

predictions = log_model.predict(telco_prepared)

confusion_matrix(y_train,predictions)

recall_score(y_train, predictions, average="binary", pos_label="Yes")

precision_score(y_train, predictions,pos_label="Yes")

f1_score(y_train, predictions,pos_label="Yes")

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

rnd_clf.fit(telco_prepared,y_train)

prediction_rnd = rnd_clf.predict(telco_prepared)

confusion_matrix(y_train,prediction_rnd)

precision_score(y_train, prediction_rnd, pos_label="Yes")

recall_score(y_train, prediction_rnd, pos_label="Yes")

f1_score(y_train, prediction_rnd,pos_label="Yes")

for name , score in zip(X_train.columns,rnd_clf.feature_importances_):
    print(name," : ", score)

from sklearn.svm import SVC

svm_clf = SVC(kernel="poly", degree=3, coef0=1, C=5)

svm_clf.fit(telco_prepared,y_train)

predictions_svm = svm_clf.predict(telco_prepared)

confusion_matrix(y_train,prediction_rnd)

recall_score(y_train, predictions_svm, pos_label="Yes")

precision_score(y_train, predictions_svm, pos_label="Yes")

f1_score(y_train, predictions_svm,pos_label="Yes")
```
