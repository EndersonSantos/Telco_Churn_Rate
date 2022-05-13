# Telco_Churn_Rate

## The problem
A lot of customers are leaving the service thus decreasing the profitability of the company.
Our mean objective is to understand what are the biggest factors that lead to churn.
The solution will be used to implement new services and changing pre-existing ones in a way that increases the life time value of each customer
#### Objectives
1. Decrease the churn rate 
2. Increase Profitability
#### How
Creating an Application that predicts if a customer will churn <br>
Developing focused customer retention programs.
#### Measure of Peformance
We're going to use *precision* as the measure of performance once understand who will indeed churn is very important to oritent our retention programs for those clients

## The Data 
We are going to use Telco Customer Churn data available on kaggle under the licence of IBM Sample Data Sets. The data can be found [here](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
#### Content 
- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

## Tecnologies Used
1. Python: Pandas, Numpy, Scikit-Learn, Seaborn, Matplotlib, Flask
2. Jupyter Notebook
3. Power Point

## Exploratory Data Analysis 
![Has Dependents](https://user-images.githubusercontent.com/90560755/168038557-3c3ab831-d46e-4f9b-b17a-99dd20120db0.png)
![Has Partner](https://user-images.githubusercontent.com/90560755/168038573-22f8ca19-2a14-4bea-8ee4-8ef334897a61.png)
![Contract Type Churn Rate](https://user-images.githubusercontent.com/90560755/168038635-5fb11373-3174-42fb-b166-4bfb6c734efa.png)
![Total Charges vs Tenure hued by Contract Type](https://user-images.githubusercontent.com/90560755/168038671-f36e8b0d-1020-4f6b-be21-2b461ed91858.png)
![Eletronic Churn](https://user-images.githubusercontent.com/90560755/168038679-f7071892-bba4-40b8-8ec1-63b75284eb7e.png)
![Churn Rate Fiber Optic](https://user-images.githubusercontent.com/90560755/168038695-72e1c4f2-058c-414f-8f73-2f521f91c95f.png)

#### Key Takeaways
- Longer the client is with the company less likely is to chunn
- The average period of time of each client is 32 months
- The clients that has higher charges is more likely to churn
- The clients who has long contracts tend to be longer with the company and the clients with a monthly contract tend to churn almost 3 times more
- 45% of the client that has an eletronic check churn
- Client that is not a Senior Citizen tend to churn more
- Single clients without any dependent also tend to churn more

## Model Building
<b>Logistic Regression</b> - 65% of precision <br>
<b>Random Forest</b> - 71% of precision <br>
<b>Support Vector Machines</b> - 74% of precision <br>

So we choose the SVM model <br>
The model has 71% of precision on the testset

## Flask App
![Churn Mode](https://user-images.githubusercontent.com/90560755/168040882-8ac77b96-362c-488e-9c97-db2a8dc314ed.png)
![Churn Model ](https://user-images.githubusercontent.com/90560755/168040891-b97a311c-77eb-40cf-8975-9f620798b92c.png) <br><br>
We give the information about the customer and the model will predict if the customer will churn
