import pandas as pd
import numpy as np
from wrangle import ADD_ONS
from IPython.display import Markdown as md
from sklearn.metrics import ConfusionMatrixDisplay,precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def prep_for_modeling(train:pd.DataFrame,validate:pd.DataFrame, test:pd.DataFrame):
    '''prep_for_modeling takes the train, validate, and test DataFrames and returns
    a tuple of the features and target of each DataFrame.'''
    model_features = ['add_ons','tenure','internet_service_type','tech_support']
    model_target = 'churn'
    return train[model_features],train[model_target],validate[model_features],validate[model_target],\
        test[model_features],test[model_target]

def get_baseline_precision(train:pd.DataFrame, mode:int):
    '''get_baseline_predictions takes the trainning dataset and the DataFrame mode and returns
    the baseline precision score.'''
    mode_arr = [mode for x in range(train.shape[0])]
    return precision_score(train.churn,mode_arr)

def get_decision_tree(train_x:pd.DataFrame,train_y:pd.DataFrame,\
    valid_x:pd.DataFrame, valid_y:pd.DataFrame)->md:
    '''get_decision_tree takes in the features and target of both the 
    train and validate DataFrames, trains a DecisionTreeClassifier
    on the train dataset, and returns a IPython.display.Markdown
    object with the precision scores for both the train and validate datasets'''
    tree = DecisionTreeClassifier(max_depth=5)
    tree.fit(train_x,train_y)
    train_predict = tree.predict(train_x)
    precision_train = precision_score(train_y, train_predict)
    valid_predict = tree.predict(valid_x)
    precision_valid = precision_score(valid_y, valid_predict)
    ret_str = f'### Decision tree precision on `train`: {precision_train}.\
        \n\n ### Decision tree precision on `validate`: {precision_valid}.'
    return md(ret_str)

def get_rf(train_x:pd.DataFrame,train_y:pd.DataFrame,\
    valid_x:pd.DataFrame, valid_y:pd.DataFrame)->md:
    '''get_rf takes in the features and target of both the train and validate
    datasets respectively, trains a RandomForestClassifier on the train dataset,
    and returns an IPython.display.Markdown object with the precision scores
    of both the train and validate datasets.'''
    rf = RandomForestClassifier(min_samples_leaf=100,max_depth=6)
    rf.fit(train_x,train_y)
    train_predict = rf.predict(train_x)
    precision_train = precision_score(train_y, train_predict)
    valid_predict = rf.predict(valid_x)
    precision_valid = precision_score(valid_y, valid_predict)
    ret_str = f'### Random Forest precision on `train`: {precision_train}.\
        \n\n ### Random Forest precision on `validate`: {precision_valid}.'
    return md(ret_str)

def get_knn(train_x:pd.DataFrame,train_y:pd.DataFrame,\
    valid_x:pd.DataFrame, valid_y:pd.DataFrame)->md:
    '''get_knn takes in the features and target of both the train and validate
    datasets respectively, trains a KNeighborClassifier on the train dataset,
    and returns an IPython.display.Markdown object with the precision scores
    of both the train and validate datasets.'''
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(train_x,train_y)
    train_predict = knn.predict(train_x)
    precision_train = precision_score(train_y, train_predict)
    valid_predict = knn.predict(valid_x)
    precision_valid = precision_score(valid_y, valid_predict)
    ret_str = f'### K Nearest Neighbors precision on `train`: {precision_train}.\
        \n\n ### K Nearest Neighbors precision on `validate`: {precision_valid}.'
    return md(ret_str)
def rf_on_test(train_x,train_y,test_x,test_y)->md:
    '''rf_on_test takes in the features and targets of the train and test DataFrames,
    trains a RandomForestClassifier model on the training data, predicts the 
    outcome on the test data set, and returns an IPython.display.Markdown objedt
    with the precision score on the test data set.'''
    rf = RandomForestClassifier(min_samples_leaf=100, max_depth=6)
    rf.fit(train_x, train_y)
    test_predict = rf.predict(test_x)
    precision_test = precision_score(test_y,test_predict)
    return md(f'### Random Forest precision on `validate`: {precision_test}')
