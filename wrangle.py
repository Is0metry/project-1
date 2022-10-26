import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os, re
from env import get_db_url


# CAT_COLS is a constant list of all columns which contain boolean values. Primarily used in clean_rows algorithm, but also used in exploration
CAT_COLS = ['partner','churn','multiple_lines','online_security','online_backup','device_protection',\
'tech_support','streaming_tv','streaming_movies','paperless_billing','dependents','phone_service']
CONTRACT_KEY = ['Month-to-month','One year','Two year']
IST_KEY = ['DSL','Fiber optic','None']
PAYMENT_KEY = ['Bank Transfer (automatic)','Credit Card (automatic)','Electronic Check','Mailed Check']

def clean_data_path(filename):
    if not filename.startswith('data/'):
        filename = 'data/' + filename
    if not filename.endswith('.csv'):
        filename = filename + '.csv'
    return filename
def build_dataframe(query,database,filename=''):
    if filename == '':
        filename += 'data/' + database + '.csv'
    else:
        filename = clean_data_path(filename)
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    url = get_db_url(database)
    df = pd.read_sql(query, url)
    df.to_csv(filename,index=False)
    return df

def tvt_split(df:pd.DataFrame,stratify:str = None,test_split:float = .2,validate_split:int = .3):
    '''This function takes a pandas DataFrame as well as either a string or pd.Series and returns a train, validate and test split of the DataFame'''
    strat = df[stratify]
    train_validate, test = train_test_split(df,test_size=test_split,random_state=123,stratify=strat)
    strat = train_validate[stratify]
    train, validate = train_test_split(train_validate,test_size=validate_split,random_state=123,stratify=strat)
    return train,validate,test
def get_telco_data():
    query = '''
    SELECT * FROM customers
    JOIN contract_types USING(contract_type_id)
    JOIN internet_service_types USING(internet_service_type_id)
    JOIN payment_types USING(payment_type_id)
    '''
    return build_dataframe(query, 'telco_churn')
def clean_rows(row:pd.Series):
    for c in CAT_COLS:
        if re.search('N|n',row[c]) is not None:
            row[c] = np.uint8(0)
        else:
            row[c] = np.uint8(1)
    row.gender = 1 if row.gender == 'Male' else 0
    row.total_charges = row.total_charges if row.tenure > 0 else 0
    contract_type = ['M','O','T']
    for i,c in enumerate(contract_type):
        if row.contract_type.startswith(c):
            row.contract_type = np.uint8(i)
            break
    internet_service_type = ['D','F','N']
    for i,c in enumerate(internet_service_type):
        if row.internet_service_type.startswith(c):
            row.internet_service_type = np.uint8(i)
            break
    payment_type = ['B','C','E','M']
    for i,c in enumerate(payment_type):
        if row.payment_type.startswith(c):
            row.payment_type = np.uint8(i)
            break
    return row
def prep_telco(telco_df:pd.DataFrame):
    telco_df = telco_df.drop(['payment_type_id','internet_service_type_id','contract_type_id','customer_id'],axis=1)
    telco_df = telco_df.apply(clean_rows,axis='columns')
    telco_df.total_charges = pd.to_numeric(telco_df.total_charges)
    telco_df.gender = telco_df.gender.astype(np.uint8)
    telco_df.payment_type = telco_df.payment_type.astype(np.uint8)
    telco_df.senior_citizen = telco_df.senior_citizen.astype(np.uint8)
    telco_df.tenure = telco_df.tenure.astype(np.uint8)
    return tvt_split(telco_df, 'churn')

if __name__ == '__main__':
    telco_df = get_telco_data()
    print(telco_df.payment_type.value_counts())
    train, validate,test = prep_telco(telco_df)