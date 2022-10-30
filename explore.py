from traceback import format_tb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown
from scipy import stats
from IPython.display import Markdown as md
from wrangle import ADD_ONS, CAT_COLS, IST_KEY, CONTRACT_KEY, PAYMENT_KEY


def tech_support_vs_churn(train:pd.DataFrame):
    fiber = train[(train.internet_service_type == 1)]
    dsl = train[train.internet_service_type == 1]
    plt.suptitle('Fiber Optic customer churn with & Without Tech Support')
    plt.subplot(2,2,1)
    plt.title('FiOS With TS')
    plt.pie(fiber[fiber.tech_support > 0].churn.value_counts())
    plt.legend(['Churned','Didn\'t churn'],loc='lower left')
    plt.subplot(2,2,2)
    plt.title('FiOS W/o TS')
    plt.pie(fiber[fiber.tech_support == 0].churn.value_counts())
   
def chi_squared(train:pd.DataFrame):
    df = train[(train.internet_service_type > 0)]
    observed = pd.crosstab(df.churn, df.tech_support)
    chi2, p, degf, expected = stats.chi2_contingency(observed) 
    return md(f'$\chi^2:{chi2:.4f}$\n\n $p:{p}$')
def churn_by_ist(train:pd.DataFrame):
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
    plt.title('Add-ons vs Tenure')
    plt.xlabel('No. of Add-ons')
    sns.barplot(data=train,x='add_ons',y='tenure')
def pearson_test(train:pd.DataFrame)->md:
    x = train.add_ons
    y = train.tenure
    r,p = stats.pearsonr(x,y)
    return md(f'$ r = {r}$\n\n$p = {p}$')
def monthly_vs_total(train:pd.DataFrame):
    sns.countplot(data=train,x='churn',hue='is_male')
    plt.xticks([0,1],['Did Not Churn','Churned'])
    plt.legend(['Female','Male'])
def add_ons_w_churn(train:pd.DataFrame):
    sns.countplot(data=train,x='add_ons',hue='churn')