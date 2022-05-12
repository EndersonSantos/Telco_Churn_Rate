# Telco Exploratory Data Analysis


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
df_ = df.copy()
```


```python
df_.head()
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
      <th>customerID</th>
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
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.5</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7043 entries, 0 to 7042
    Data columns (total 21 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   customerID        7043 non-null   object 
     1   gender            7043 non-null   object 
     2   SeniorCitizen     7043 non-null   int64  
     3   Partner           7043 non-null   object 
     4   Dependents        7043 non-null   object 
     5   tenure            7043 non-null   int64  
     6   PhoneService      7043 non-null   object 
     7   MultipleLines     7043 non-null   object 
     8   InternetService   7043 non-null   object 
     9   OnlineSecurity    7043 non-null   object 
     10  OnlineBackup      7043 non-null   object 
     11  DeviceProtection  7043 non-null   object 
     12  TechSupport       7043 non-null   object 
     13  StreamingTV       7043 non-null   object 
     14  StreamingMovies   7043 non-null   object 
     15  Contract          7043 non-null   object 
     16  PaperlessBilling  7043 non-null   object 
     17  PaymentMethod     7043 non-null   object 
     18  MonthlyCharges    7043 non-null   float64
     19  TotalCharges      7043 non-null   object 
     20  Churn             7043 non-null   object 
    dtypes: float64(1), int64(2), object(18)
    memory usage: 1.1+ MB
    


```python
df_.isnull().sum()
```




    customerID          0
    gender              0
    SeniorCitizen       0
    Partner             0
    Dependents          0
    tenure              0
    PhoneService        0
    MultipleLines       0
    InternetService     0
    OnlineSecurity      0
    OnlineBackup        0
    DeviceProtection    0
    TechSupport         0
    StreamingTV         0
    StreamingMovies     0
    Contract            0
    PaperlessBilling    0
    PaymentMethod       0
    MonthlyCharges      0
    TotalCharges        0
    Churn               0
    dtype: int64




```python
df_.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SeniorCitizen</th>
      <td>7043.0</td>
      <td>0.162147</td>
      <td>0.368612</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>tenure</th>
      <td>7043.0</td>
      <td>32.371149</td>
      <td>24.559481</td>
      <td>0.00</td>
      <td>9.0</td>
      <td>29.00</td>
      <td>55.00</td>
      <td>72.00</td>
    </tr>
    <tr>
      <th>MonthlyCharges</th>
      <td>7043.0</td>
      <td>64.761692</td>
      <td>30.090047</td>
      <td>18.25</td>
      <td>35.5</td>
      <td>70.35</td>
      <td>89.85</td>
      <td>118.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_.describe(include="object").T
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
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>customerID</th>
      <td>7043</td>
      <td>7043</td>
      <td>8204-YJCLA</td>
      <td>1</td>
    </tr>
    <tr>
      <th>gender</th>
      <td>7043</td>
      <td>2</td>
      <td>Male</td>
      <td>3555</td>
    </tr>
    <tr>
      <th>Partner</th>
      <td>7043</td>
      <td>2</td>
      <td>No</td>
      <td>3641</td>
    </tr>
    <tr>
      <th>Dependents</th>
      <td>7043</td>
      <td>2</td>
      <td>No</td>
      <td>4933</td>
    </tr>
    <tr>
      <th>PhoneService</th>
      <td>7043</td>
      <td>2</td>
      <td>Yes</td>
      <td>6361</td>
    </tr>
    <tr>
      <th>MultipleLines</th>
      <td>7043</td>
      <td>3</td>
      <td>No</td>
      <td>3390</td>
    </tr>
    <tr>
      <th>InternetService</th>
      <td>7043</td>
      <td>3</td>
      <td>Fiber optic</td>
      <td>3096</td>
    </tr>
    <tr>
      <th>OnlineSecurity</th>
      <td>7043</td>
      <td>3</td>
      <td>No</td>
      <td>3498</td>
    </tr>
    <tr>
      <th>OnlineBackup</th>
      <td>7043</td>
      <td>3</td>
      <td>No</td>
      <td>3088</td>
    </tr>
    <tr>
      <th>DeviceProtection</th>
      <td>7043</td>
      <td>3</td>
      <td>No</td>
      <td>3095</td>
    </tr>
    <tr>
      <th>TechSupport</th>
      <td>7043</td>
      <td>3</td>
      <td>No</td>
      <td>3473</td>
    </tr>
    <tr>
      <th>StreamingTV</th>
      <td>7043</td>
      <td>3</td>
      <td>No</td>
      <td>2810</td>
    </tr>
    <tr>
      <th>StreamingMovies</th>
      <td>7043</td>
      <td>3</td>
      <td>No</td>
      <td>2785</td>
    </tr>
    <tr>
      <th>Contract</th>
      <td>7043</td>
      <td>3</td>
      <td>Month-to-month</td>
      <td>3875</td>
    </tr>
    <tr>
      <th>PaperlessBilling</th>
      <td>7043</td>
      <td>2</td>
      <td>Yes</td>
      <td>4171</td>
    </tr>
    <tr>
      <th>PaymentMethod</th>
      <td>7043</td>
      <td>4</td>
      <td>Electronic check</td>
      <td>2365</td>
    </tr>
    <tr>
      <th>TotalCharges</th>
      <td>7043</td>
      <td>6531</td>
      <td></td>
      <td>11</td>
    </tr>
    <tr>
      <th>Churn</th>
      <td>7043</td>
      <td>2</td>
      <td>No</td>
      <td>5174</td>
    </tr>
  </tbody>
</table>
</div>



#### Total Charge was formated as an object beacause it contains space caracters  <br> So we need to format as float, first we gonna replace the spaces for 0 and then write a lambda function to convert into a float 


```python
df_["TotalCharges"].replace(" ", "0", inplace=True)
```


```python
df_["TotalCharges"] = df_["TotalCharges"].apply(lambda x: float(x))
```

#### Analysing the distribution and correlation of the numerical features


```python
sns.pairplot(df_)
```




    <seaborn.axisgrid.PairGrid at 0x186f55000a0>




    
![png](output_14_1.png)
    



```python
df_.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SeniorCitizen</th>
      <td>7043.0</td>
      <td>0.162147</td>
      <td>0.368612</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>tenure</th>
      <td>7043.0</td>
      <td>32.371149</td>
      <td>24.559481</td>
      <td>0.00</td>
      <td>9.00</td>
      <td>29.00</td>
      <td>55.00</td>
      <td>72.00</td>
    </tr>
    <tr>
      <th>MonthlyCharges</th>
      <td>7043.0</td>
      <td>64.761692</td>
      <td>30.090047</td>
      <td>18.25</td>
      <td>35.50</td>
      <td>70.35</td>
      <td>89.85</td>
      <td>118.75</td>
    </tr>
    <tr>
      <th>TotalCharges</th>
      <td>7043.0</td>
      <td>2279.734304</td>
      <td>2266.794470</td>
      <td>0.00</td>
      <td>398.55</td>
      <td>1394.55</td>
      <td>3786.60</td>
      <td>8684.80</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_["TotalCharges"].mode()
```




    0     0.0
    1    20.2
    dtype: float64



#### total charge is the monthly charge times the tenure. Since some clients do not even stay one month with the company the tenure is zero so the total charge should be only the monthly that the client was charged <br> So we gonna write a function that multiply the tenure and the monthly rate and in the case that the tenure is zero return the value of the monthly rate that was the only charge that the client payed for 




```python
def total_charge(tenure,monthlyCharge, totalCharges):
    if totalCharges != float(0):
        return totalCharges
    else:
        return monthlyCharge
```


```python
df_["TotalCharges"] = df_[["tenure", "MonthlyCharges","TotalCharges"]].apply(lambda df_: total_charge(df_["tenure"],df_["MonthlyCharges"],df_["TotalCharges"]), axis=1)
```

#### Looking for outiliers


```python
sns.boxplot(df_["TotalCharges"])
```

    C:\Users\ewill\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <AxesSubplot:xlabel='TotalCharges'>




    
![png](output_21_2.png)
    



```python
sns.boxplot(df_["MonthlyCharges"])
```

    C:\Users\ewill\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <AxesSubplot:xlabel='MonthlyCharges'>




    
![png](output_22_2.png)
    



```python
  sns.boxplot(df_["tenure"])
```

    C:\Users\ewill\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <AxesSubplot:xlabel='tenure'>




    
![png](output_23_2.png)
    


#### Analysing the Tenure


```python
df_["tenure"].mean()
```




    32.37114865824223




```python
def tenure(tenure_):
    if tenure_ <12:
        return "less than 1 year"
    elif tenure_>=12 and tenure_<24:
        return "1 to 2 years"
    elif tenure_>=24 and tenure_ <48:
            return "2 to 4 years"
    else:
        return "more than 4 year"
```


```python
df_["tenure_cat"] = df_["tenure"].apply(tenure)
```


```python
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_, y="TotalCharges", x="MonthlyCharges", hue="tenure_cat", alpha=0.5, palette="viridis")
handles, labels = plt.gca().get_legend_handles_labels()
order = [3,1,2,0]
plt.legend([handles[i] for i in order], [labels[i] for i in order],fontsize=14)
plt.title("Total charges vs Monthly Charges divided by Tenure", fontsize=16)
plt.ylabel("Total Charges")
plt.xlabel("Monthly Charges")
```




    Text(0.5, 0, 'Monthly Charges')




    
![png](output_28_1.png)
    


#### Visualizing the Data


```python
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_, y="TotalCharges", x="MonthlyCharges", hue="Churn", alpha=0.7, palette="viridis")
handles, labels = plt.gca().get_legend_handles_labels()
plt.title("Total charges vs Monthly Charges divided by Churn", fontsize=16)
plt.ylabel("Total Charges")
plt.xlabel("Monthly Charges")
plt.legend(fontsize=20)
save_fig("Total charges vs Monthly Charges divided by Churn")
```

    Saving figure Total charges vs Monthly Charges divided by Churn
    


    
![png](output_30_1.png)
    



```python
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_, y="TotalCharges", x="tenure", hue="Churn",alpha=0.7,palette="viridis")
plt.title("Total charges vs Tenure divided by Churn", fontsize=16)
plt.ylabel("Total Charges")
plt.xlabel("Tenure")
plt.legend(fontsize=20)
```




    <matplotlib.legend.Legend at 0x186f6415a30>




    
![png](output_31_1.png)
    



```python
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_, y="tenure", x="MonthlyCharges", hue="Churn",alpha=0.7, palette="viridis")
plt.title("Tenure vs Monthly Charge divided by Churn", fontsize=16)
plt.ylabel("Tenure")
plt.xlabel("Monthly Charges")
plt.legend(loc=(0.9,0.05),fontsize=20)
save_fig("Tenure Vs Monthly Charge")
```

    Saving figure Tenure Vs Monthly Charge
    


    
![png](output_32_1.png)
    



```python
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_, y="TotalCharges", x="tenure", hue="Contract", alpha=0.5, palette="viridis")
handles, labels = plt.gca().get_legend_handles_labels()
order = [2, 1, 0]
plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=14)
plt.title("Total Charges vs Tenure Divided by Contract Type", fontsize=16)
plt.ylabel("Total Charges")
save_fig("Total Charges vs Tenure hued by Contract Type")
```

    Saving figure Total Charges vs Tenure hued by Contract Type
    


    
![png](output_33_1.png)
    



```python
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_, y="TotalCharges", x="MonthlyCharges", hue="PaperlessBilling",alpha=0.7, palette="viridis")
plt.legend(fontsize=20)
plt.ylabel("Total Charges")
plt.xlabel("Monthly Charges")
plt.legend(loc=(0.1,0.7), fontsize=20)
plt.title("Total Charges vs Monthly Charges divided by Paperless Billing", fontsize=16)
```




    Text(0.5, 1.0, 'Total Charges vs Monthly Charges divided by Paperless Billing')




    
![png](output_34_1.png)
    


Dos que fizeram churn analysis 


```python
df_churn_yes = df_[df_["Churn"] == "Yes"]
```


```python
df_churn_no = df_[df_["Churn"] == "No"]
```


```python
df_churn_yes["PaperlessBilling"].value_counts()
```




    Yes    1400
    No      469
    Name: PaperlessBilling, dtype: int64




```python
df_churn_no["PaperlessBilling"].value_counts()
```




    Yes    2771
    No     2403
    Name: PaperlessBilling, dtype: int64




```python
df_churn_yes["Partner"].value_counts()
```




    No     1200
    Yes     669
    Name: Partner, dtype: int64



#### Analysis of the customer profile

Percentage of churn rate given the gender


```python
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_, y="TotalCharges", x="MonthlyCharges", hue="gender", alpha=0.5, palette="viridis")
plt.legend(fontsize=20)
plt.ylabel("Total Charges")
plt.xlabel("Monthly Charges")
plt.legend(loc=(0.1,0.7), fontsize=20)
plt.title("Total Charges vs Monthly Charges divided by Gender", fontsize=16)
```




    Text(0.5, 1.0, 'Total Charges vs Monthly Charges divided by Gender')




    
![png](output_43_1.png)
    



```python
df_[df_["Churn"] == "Yes"].value_counts("gender").sort_index()/df_["gender"].value_counts().sort_index()
```




    gender
    Female    0.269209
    Male      0.261603
    dtype: float64




```python
plt.figure(figsize=(10,6))
sns.countplot(df_["gender"], hue=df_["Churn"], palette="viridis")
plt.text(0.11,400,"27%", fontsize=22)
plt.text(1.13,400,"26%", fontsize=22)
plt.title("Churn Rate vs Gender", fontsize=22)
plt.ylabel("Number of Clients")
plt.legend(fontsize=20)
```

    C:\Users\ewill\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <matplotlib.legend.Legend at 0x186f7970760>




    
![png](output_45_2.png)
    



```python
df_[df_["Churn"] == "Yes"].value_counts("SeniorCitizen").sort_index()/df_["SeniorCitizen"].value_counts().sort_index()
```




    SeniorCitizen
    0    0.236062
    1    0.416813
    dtype: float64




```python
plt.figure(figsize=(10,6))
sns.countplot(df_["SeniorCitizen"], hue=df_["Churn"], palette="viridis")
plt.text(0.13,600,"24%", fontsize=22)
plt.text(1.13,150,"41%", fontsize=22)
plt.title("Churn Rate vs Citizen", fontsize=22)
plt.ylabel("Number of Clients")
plt.xlabel("Senior Citizen")
plt.legend(fontsize=20)
```

    C:\Users\ewill\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <matplotlib.legend.Legend at 0x186f7746250>




    
![png](output_47_2.png)
    



```python
df_[df_["Churn"] == "Yes"].value_counts("Partner").sort_index()/df_["Partner"].value_counts().sort_index()
```




    Partner
    No     0.329580
    Yes    0.196649
    dtype: float64




```python
plt.figure(figsize=(10,6))
sns.countplot(df_["Partner"], hue=df_["Churn"],order=["No","Yes"], palette="viridis")
plt.text(0.13,600,"33%", fontsize=22)
plt.text(1.13,300,"20%", fontsize=22)
plt.title("Churn Rate vs Has Partner", fontsize=22)
plt.ylabel("Number of Clients")
plt.legend(fontsize=20)
save_fig("Has Partner")
```

    C:\Users\ewill\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    

    Saving figure Has Partner
    


    
![png](output_49_2.png)
    



```python
df_[df_["Churn"] == "Yes"].value_counts("Dependents").sort_index()/df_["Dependents"].value_counts().sort_index()
```




    Dependents
    No     0.312791
    Yes    0.154502
    dtype: float64




```python
plt.figure(figsize=(10,6))
sns.countplot(df_["Dependents"], hue=df_["Churn"], palette="viridis")
plt.text(0.13,650,"31%", fontsize=22)
plt.text(1.13,100,"16%", fontsize=22)
plt.title("Churn Rate vs Has Dependents", fontsize=22)
plt.ylabel("Number of Clients")
plt.legend(fontsize=20)
save_fig("Has Dependents")
```

    C:\Users\ewill\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    

    Saving figure Has Dependents
    


    
![png](output_51_2.png)
    


## Analysis given Contract Nuances


```python
df_[df_["Churn"] == "Yes"].value_counts("Contract").sort_index()/df_["Contract"].value_counts().sort_index()
```




    Contract
    Month-to-month    0.427097
    One year          0.112695
    Two year          0.028319
    dtype: float64




```python
plt.figure(figsize=(10,6))
sns.countplot(df_["Contract"], hue=df_["Churn"], palette="viridis")
plt.text(0.1,650,"43%", fontsize=22)
plt.text(1.1,200,"11%", fontsize=22)
plt.text(2.1,100,"2.8%", fontsize=22)
plt.title("Churn Rate vs Contract Type", fontsize=22)
plt.ylabel("Number of Clients")
plt.legend(fontsize=20);
save_fig("Contract Type Churn Rate")
```

    C:\Users\ewill\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    

    Saving figure Contract Type Churn Rate
    


    
![png](output_54_2.png)
    



```python
df_[df_["Churn"] == "Yes"].value_counts("PaymentMethod").sort_index()/df_["PaymentMethod"].value_counts().sort_index()
```




    PaymentMethod
    Bank transfer (automatic)    0.167098
    Credit card (automatic)      0.152431
    Electronic check             0.452854
    Mailed check                 0.191067
    dtype: float64




```python
plt.figure(figsize=(12,6))
sns.countplot(df_["PaymentMethod"], hue=df_["Churn"], palette="viridis")
plt.text(0.07,500,"45%", fontsize=22)
plt.text(1.07,120,"19%", fontsize=22)
plt.text(2.07,100,"18%", fontsize=22)
plt.text(3.07,100,"15%", fontsize=22)
plt.title("Churn Rate vs Payment Method", fontsize=22)
plt.ylabel("Number of Clients")
plt.legend(fontsize=20,loc=(1.01,0.5))
plt.xlabel("Payment Method")
save_fig("Eletronic Churn")
```

    C:\Users\ewill\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    

    Saving figure Eletronic Churn
    


    
![png](output_56_2.png)
    



```python
df_["PaymentMethod"].value_counts()
```




    Electronic check             2365
    Mailed check                 1612
    Bank transfer (automatic)    1544
    Credit card (automatic)      1522
    Name: PaymentMethod, dtype: int64




```python
df_[df_["Churn"] == "Yes"].value_counts("PaperlessBilling").sort_index()/df_["PaperlessBilling"].value_counts().sort_index()
```




    PaperlessBilling
    No     0.163301
    Yes    0.335651
    dtype: float64




```python
plt.figure(figsize=(10,6))
sns.countplot(df_["PaperlessBilling"], hue=df_["Churn"],order=["No","Yes"], palette="viridis")
plt.text(0.12,200,"16%", fontsize=22)
plt.text(1.13,550,"34%", fontsize=22)
plt.title("Churn Rate vs Paperless Billing", fontsize=22)
plt.ylabel("Number of Clients")
plt.legend(fontsize=20)
plt.xlabel("Paperless Billing");
```

    C:\Users\ewill\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_59_1.png)
    


# -------------------------------------------------


```python
df_["PaperlessBilling"].value_counts()
```




    Yes    4171
    No     2872
    Name: PaperlessBilling, dtype: int64




```python
df_internet = df_[(df_["InternetService"] == "DSL") | (df_["InternetService"] == "Fiber optic")]
```


```python
df_internet[df_internet["Churn"] == "Yes"].value_counts("InternetService").sort_index()/df_internet["InternetService"].value_counts().sort_index()
```




    InternetService
    DSL            0.189591
    Fiber optic    0.418928
    dtype: float64




```python
plt.figure(figsize=(10,6))
sns.countplot(df_internet["InternetService"], hue=df_["Churn"], palette="viridis")
plt.text(0.12,200,"19%", fontsize=22)
plt.text(1.13,550,"42%", fontsize=22)
plt.title("Churn Rate vs Internet Service", fontsize=22)
plt.ylabel("Number of Clients")
plt.legend(fontsize=20)
plt.xlabel("Internet Service")
save_fig("Churn Rate Fiber Optic");
```

    C:\Users\ewill\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    

    Saving figure Churn Rate Fiber Optic
    


    
![png](output_64_2.png)
    


We want to see if the total number of services has any impact on the ternure duration <br> We going to write a function that sum the number of services that each client has 


```python
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
    return count
```


```python
df_["Number_Services"] = df_[["PhoneService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV", "StreamingMovies"]].apply(lambda df_: total_services(df_["PhoneService"],df_["OnlineSecurity"],df_["OnlineBackup"],df_["DeviceProtection"],df_["TechSupport"], df_["StreamingTV"], df_["StreamingMovies"]), axis=1)
```


```python
plt.figure(figsize=(12,6))
sns.countplot(df_["Number_Services"], hue=df_["Churn"], palette="viridis")
plt.xlabel("Number of Services")
plt.ylabel("Frequency")
plt.legend(fontsize=16)
plt.title("Number of Services of Each Client Divided by Churn", fontsize=16)
```

    C:\Users\ewill\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    Text(0.5, 1.0, 'Number of Services of Each Client Divided by Churn')




    
![png](output_68_2.png)
    



```python
df_["Number_Services"].value_counts().sort_index()
```




    1      80
    2    2253
    3     996
    4    1041
    5    1062
    6     827
    7     525
    8     259
    Name: Number_Services, dtype: int64




```python
df_[df_["Churn"] == "Yes"].value_counts("Number_Services").sort_index()
```




    Number_Services
    1     35
    2    488
    3    433
    4    361
    5    289
    6    182
    7     66
    8     15
    dtype: int64




```python
df_[df_["Churn"] == "Yes"].value_counts("Number_Services").sort_index()/df_["Number_Services"].value_counts().sort_index()
```




    Number_Services
    1    0.437500
    2    0.216600
    3    0.434739
    4    0.346782
    5    0.272128
    6    0.220073
    7    0.125714
    8    0.057915
    dtype: float64



#### Correlation Among Variables


```python
df_dummies = df_.drop("customerID",axis=1)
```


```python
df_dummies = pd.get_dummies(df_dummies, drop_first=True)
```


```python
corr_churn = df_dummies.corr()["Churn_Yes"].sort_values()
corr_churn = corr_churn[:-1]
```


```python
plt.figure(figsize=(12,8))
corr_churn.plot(kind="bar")
plt.title("Correlation Among Variables", fontsize=16)
```




    Text(0.5, 1.0, 'Correlation Among Variables')




    
![png](output_76_1.png)
    


### Key Takeaways

- Longer the client is with the company less likely is to chunn  
- The average period of time of each client is 32 months 
- The clients that has higher charges is more likely to churn 
- The clients who has long contracts tend to be longer with the company and the clients with a monthly contract tend to churn almost 3 times more 
- 45% of the client that has an eletronic check churn 
- Client that is not a Senior Citizen tend to churn more 
- Single clients without any dependent also tend to churn more
