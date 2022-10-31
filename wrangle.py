import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import os, re,pydoc
from env import host, user,password


# Add-ons used for cleaning 
ADD_ONS = ['online_security','online_backup','device_protection',\
'tech_support','streaming_tv','streaming_movies','paperless_billing','multiple_lines']
'''CAT_COLS is a constant list containing all categorical features in the data frame'''
CAT_COLS = ADD_ONS + ['senior_citizen','dependents','phone_service','churn','partner']

CONTRACT_KEY = ['Month-to-month','One year',' partner','Two year']
IST_KEY = ['None','DSL','Fiber optic']
PAYMENT_KEY = ['Bank transfer (automatic)','Credit card (automatic)','Electronic check','Mailed check']
def get_db_url(database:str):
    '''get_db_url takes a str with the database to connect to and returns a formatted string
    with the hostname, username, and password imported from env.py'''
    return f'mysql+pymysql://{user}:{password}@{host}/{database}'
def clean_data_path(filename:str)->str:
    '''clean_data_path takes a string with the name of a file and modifies it to fit the desired filepath format i.e. data/*.csv'''
    if not filename.startswith('data/'):
        filename = 'data/' + filename
    if not filename.endswith('.csv'):
        filename = filename + '.csv'
    return filename
def build_dataframe(query:str,database:str,filename:str='')->pd.DataFrame:
    '''build_dataframe loads in the locally cached CSV file if it exist. otherwise, it will fetch the URL for the database, and run query 
    to load a dataframe from SQL. Returns a dataframe with selected information from query.'''
    if filename == '':
        filename = database
    filename = clean_data_path(filename)
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    url = get_db_url(database)
    df = pd.read_sql(query, url)
    df.to_csv(filename,index=False)
    return df

def tvt_split(df:pd.DataFrame,stratify:str = None,tv_split:float = .2,validate_split:int = .3):
    '''tvt_split takes a pandas DataFrame, a string specifying the variable to stratify over,
    as well as 2 floats where 0< f < 1 and returns a train, validate, and test split of the DataFame,
    split by tv_split initially and validate_split thereafter. '''
    strat = df[stratify]
    train_validate, test = train_test_split(df,test_size=tv_split,random_state=123,stratify=strat)
    strat = train_validate[stratify]
    train, validate = train_test_split(train_validate,test_size=validate_split,random_state=123,stratify=strat)
    return train,validate,test
def get_telco_data(prepped:bool = False):
    '''get_telco_data takes an optional argument prepped. Returns the encoded and cleaned telco_churn DataFrame if `prepped`
    otherwise, it returns the raw telco_df'''
    #creates query to request necessary information from telco_churn database
    query = '''
    SELECT * FROM customers
    JOIN contract_types USING(contract_type_id)
    JOIN internet_service_types USING(internet_service_type_id)
    JOIN payment_types USING(payment_type_id)
    '''
    telco_df = build_dataframe(query, 'telco_churn')
    #if prepped: prepares the dataframe
    if prepped:
        telco_df = prep_telco(telco_df)
    return telco_df
def clean_rows(row:pd.Series):
    '''clean_rows is a helper function which takes a pandas.Series which represents a row of the telco_churn DataFrame
    it first collapses the values in CAT_COLS from 3 different values (Yes, No, No phone|internet service) to a boolean.
    it then changes row.gender to be a boolean (column will later be renamed is_male to reflect this). Next, it updates
    the dtype of total_charges, changing ' ' of customers with 0 tenure to have 0 in total charges.Finally, it encodes 
    internet_service_type, payment_type, and contract_type as np.uint8 in alphabetical order. See 
    data dictionary and {col}_KEY for additional information. It then returns the modified row
    '''
    for c in CAT_COLS:
        if re.search('N|n',row[c]) is not None:
            row[c] = np.uint8(0)
        else:
            row[c] = np.uint8(1)
    row.gender = 1 if row.gender == 'Male' else 0
    row.total_charges = row.total_charges if row.tenure > 0 else 0
    for i,c in enumerate(CONTRACT_KEY):
        if row.contract_type.startswith(c):
            row.contract_type = np.uint8(i)
            break
    for i,c in enumerate(IST_KEY):
        if row.internet_service_type.startswith(c):
            row.internet_service_type = np.uint8(i)
            break
    for i,c in enumerate(PAYMENT_KEY):
        if row.payment_type.startswith(c):
            row.payment_type = np.uint8(i)
            break
    return row
    
def prep_telco(telco_df:pd.DataFrame):
    '''prep_telco prepares the telco_training data sets and performs transformations on it to prepare it for classification and modelling,
    droping 'id' types, applying clean_rows function (see above) row-wise. It converts payment_type, senior_citizen, and tenure to unsigned
    8-bit integers in order to conserve memory. Finally, it returns tvt_split of the DataFrame, returning a tuple of DataFrames for training,
    validating, and testing '''
    telco_df = telco_df.drop(['payment_type_id','internet_service_type_id','contract_type_id','customer_id'],axis=1)
    telco_df = telco_df.apply(clean_rows,axis='columns')
    telco_df.total_charges = pd.to_numeric(telco_df.total_charges)
    telco_df.gender = telco_df.gender.astype(np.uint8)
    telco_df = telco_df.rename(columns={'gender':'is_male'})
    telco_df.payment_type = telco_df.payment_type.astype(np.uint8)
    telco_df.senior_citizen = telco_df.senior_citizen.astype(np.uint8)
    telco_df.tenure = telco_df.tenure.astype(np.uint8)
    telco_df['add_ons'] = telco_df[ADD_ONS].sum(axis='columns')
    return telco_df