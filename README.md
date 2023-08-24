# telco_churn_project
# Drivers for Customer Churn 

# Project Description: 
- Customers are looking for the best services possible. We want to know what are driving factors to customers in order for them to stay or churn. Using customer data, services data, and payment data in order to identify what those drivers are. 

# Project Goals 
- Find drivers for customer churn at Telco. Why are customers churning?
- Construct a ML classification model that accurately predicts customer churn
- Present your process and findings to the lead data scientist

# The Plan 
Aquire data from SQL

Prepare data

Create Engineered columns from existing data
upset
Contract Type
Payment Type
Internet Service type


Explore data in search of drivers of Churn
- What featues may lead to the rate of churn? 
- Do monthy charges impact churn? 
- Does being a senior citizen impact who churned? If it does, do I need to change my strategy of imputation? (in other words, do those assumptions have big impact)
- Are customers with DSL more or less likely to churn?
- What month are customers most likely to churn and does that depend on their contract type?
- Is there a service that is associated with more churn than expected?
- Do customers who churn have a higher average monthly spend than those who don't?

Develop a Model to predict how likly a customer will churn 

Use drivers identified in explore to build predictive models of different types
Evaluate models on train and validate data
Select the best model based on highest accuracy
Evaluate the best model on test data
Draw conclusions


Data Dictionary 

| Feature      | Definition             |  
|--------------|------------------------|
| Contract Type | month to month, one year, or two year|
| Payment Type      | mailed check, Electronic Check, Credit Card, Bank Transfer |


Steps to reporduce 

- Clone this repo.
- Acquire the data from MySql
- Put the data in the file containing the cloned repo.
- Run notebook.

Key take aways and conclusions 
- Get customers on one to two year plans to 
- Update customer payment type and promot auto pay
- Run promotions to upgrade existing customers internet service 


Reccomendations 
To increase the skill intensity of a game add to the length of time players are able to consider their moves
Based on the data longer time controls make it less likely for a less skilled player to beat a more skilled player