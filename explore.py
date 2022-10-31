from traceback import format_tb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown
from scipy import stats
from IPython.display import Markdown as md
from wrangle import ADD_ONS, CAT_COLS, IST_KEY, CONTRACT_KEY, PAYMENT_KEY


def tech_support_vs_churn(train:pd.DataFrame)->None:
    '''tech_support_vs_churn takes the training DataFrame and shows 2
    pie charts, one showing churn proportions for users with tech support
    and the other with tech support'''
    fiber = train[(train.internet_service_type == 1)]
    plt.suptitle('Fiber Optic customer churn with & Without Tech Support')
    plt.subplot(2,2,1)
    plt.title('FiOS With TS')
    plt.pie(fiber[fiber.tech_support > 0].churn.value_counts())
    plt.legend(['Churned','Didn\'t churn'],loc='lower left')
    plt.subplot(2,2,2)
    plt.title('FiOS W/o TS')
    plt.pie(fiber[fiber.tech_support == 0].churn.value_counts())
   
def chi_squared(train:pd.DataFrame):
    '''chi_squared takes the training dataset and returns an IPython.display.Markdown
    object containg the formatted chi squared and p-value for the operation'''
    df = train[(train.internet_service_type > 0)]
    observed = pd.crosstab(df.churn, df.tech_support)
    chi2, p, degf, expected = stats.chi2_contingency(observed) 
    return md(f'$\chi^2:{chi2:.4f}$\n\n $p:{p}$')
def churn_by_ist(train:pd.DataFrame):
    '''churn_by_ist takes the train DataFrame and creates
    3 pie charts indicating users who churned and did not churn
    separated by internet_service_type.'''
    none = train[(train.internet_service_type == 0)].churn.value_counts().sort_index()
    dsl = train[(train.internet_service_type == 1)].churn.value_counts().sort_index()
    fib_op = train[(train.internet_service_type == 2)].churn.value_counts().sort_index()
    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1)
    plt.title('None')
    plt.pie(none)
    plt.subplot(1,3,2)
    plt.pie(fib_op)
    plt.title('Fiber Optic')
    plt.subplot(1,3,3)
    plt.pie(dsl)
    plt.title('DSL')
    plt.legend(['Churned','Didn\'t churn'],loc='lower right')
    plt.show()
def add_ons_vs_tenure(train:pd.DataFrame):
    '''add ons takes the training dataset and shows a barplot of number of add-ons
    vs the average customer tenure'''
    plt.title('Add-ons vs Tenure')
    plt.xlabel('No. of Add-ons')
    sns.barplot(data=train,x='add_ons',y='tenure')
def pearson_test(train:pd.DataFrame)->md:
    '''pearson_test takes the training DataFrame and performs a pearson correlation coefficient
    operation on the number of add_ons vs. the average customer's tenure. It returns
    an IPython.display.Markdown object with the r and p values.'''
    x = train.add_ons
    y = train.tenure
    r,p = stats.pearsonr(x,y)
    return md(f'$ r = {r}$\n\n$p = {p}$')
def gender_vs_churn(train:pd.DataFrame):
    '''gender_vs_churn takes the training DataFrame and shows a plot
    showing churn counts by gender'''
    sns.countplot(data=train,x='churn',hue='is_male')
    plt.xticks([0,1],['Did Not Churn','Churned'])
    plt.legend(['Female','Male'])
def add_ons_w_churn(train:pd.DataFrame):
    '''add_ons_w_churn takes the training database and shows a seaborn plot
    showing the number of customers with a certain number of add_ons grouped
    by churn.'''
    sns.countplot(data=train,x='add_ons',hue='churn')