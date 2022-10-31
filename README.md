# Goals
* Determine the driving factors of churn within Telco
* Use those drivers to build a model of that churn
* Try to predict whether a customer is likely to churn

# Plan
- Acquire data from database or [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Prepare data
	- Create engineered features using existing data:
		- add_ons
- Explore data in search of drivers of churn
	- Answer the following initial questions
		- Does internet service type affect churn?
		- Do internet customers with tech support churn less?
		- Does gender affect churn?
		- Does number of add-ons affect churn?
- Model data
    - Use drivers identified in explore to build predictive models of different types
    - Evaluate models on train and validate data
    - Select the best model based on highest accuracy
    - Evaluate the best model on test data
- Draw conclusions

# Data Dictionary
**Feature** | **Definition**
---|---
 **is_male** | Encoded `gender` column from original telco_churn data. 1 if customer is male, 0 otherwise.
 **senior_citizen**|1 if the customer is over 65, 0 otherwise.
 **partner** | 1 if the customer has a spouse/partner, 0 otherwise
 **dependent** | 1 if the customer has dependents, 0 otherwise
 **tenure** | how many months the customer has been with Telco
**phone_service** | 1 if the user has phone service, 0 otherwise
**multiple_lines** | 1 if the user has phone service with multiple lines, 0 otherwise.
**online_security** | 1 if the user has internet service with online security, 0 otherwise.
**online_backup** | 1 if the suer has internet service with online backup, 0 otherwise.
**device_protection** | 1 if the user has internet service with data online protection, 0 otherwise. 
**tech_support** | 1 if the suer has internet service with technical support, 0 otherwise.
**streaming_tv** | 1 if the user has internet service with streaming television, 0 otherwise
**streaming_movies** | 1 if user has internet access with streaming movies, 0 otherwise
**paperless_billing** | 1 if the user is enrolled in paperless billing, 0 otherwise
**monthly_charges** | float containing the total monthly charge of user's service 
**total_charges** | total charges over the lifetime of the user's account
**churn** | whether or not the customer has churned.
**contract_type** | encoded type of customer's contract type (see relevant value dictionary sections below)
**internet_service\_type** | encoded type of internet service type (see relevant value dictionary below)
**payment_type** | encoded payment type (see relevant value dictionary below)
**add_ons** | number of add-ons associated with customer's account
# Value Dictionaries
*Note: values are encoded in alphabetical order*
## contract_type
encoded value |actual value
---|---
0 | Month To Month
1 | One Year contract
2 | Two Year Contract
## internet_service\_type:
encoded value |actual value
---|---
0 | None
1 | Fiber Optic
2 | DSL

## payment_type:
encoded value |actual value
---|---
0|Bank Transfer (automatic)
1|Credit Card (automatic
2| Electronic check
3| Mailed check

# Steps to Reproduce:
1. Clone this repo
2. provide env.py file with hostname, username, and password, and database for telco data **OR** download `.csv` file from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
3. Place `.csv` file in `data/telco.csv`
3. run the notebook

# Conclusions:
## Exploration:
- Churn impacts roughtly a quarter of current customer base
- Fiber optic customers churn the most
    - This appears to be mitigated with tech support
- More add-ons = more tenure
- customer's gender does not appear to have an effect on churn

## Modeling
**The final model significantly outperformed baseline in terms of precision**

## Recommendations:
- Bundle add-ons, as a greater number of add-ons appears to reduce customer churn
- Offer technical support for all internet users, or, more precisely, fiber optic users

## Next Steps:
- Investigate individual add-ons for correlation with customer churn
- Investigate demographics in order to better target customers for long-term growth.