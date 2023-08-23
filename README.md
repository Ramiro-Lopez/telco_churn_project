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
rating_difference
game_rating
lower_rated_white
time_control_group

Explore data in search of drivers of upsets
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

Steps to reporduce 

Key take aways and conclusions 

Reccomendations 
