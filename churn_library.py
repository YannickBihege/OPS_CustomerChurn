# library doc string


# import libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns;

sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import seaborn as sns

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import logging


def import_data(pth):
    '''
   returns dataframe for the csv found at pth

   input:
           pth: a path to the csv
   output:
           df: pandas dataframe
   '''

    df = pd.read_csv(pth)
    logging.info(df.shape)
    logging.info(df.isnull().sum())
    return df


def perform_eda(df):
    '''
        Performs exploratory data analysis on the dataframe.

        df : DataFrame
            Input dataframe for EDA.

        returns : None
            Saves the plots from EDA to disk.
        '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig('Churn.png')

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig('Customer Age.png')

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('Marital status.png')
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('Total Trans CT')

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('Heat map')

    logging.info('Saving images terminated successfully!')


cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

quant_columns = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]


def encoder_helper(df, category_lst, response):
    '''
   helper function to turn each categorical column into a new column with
   propotion of churn for each category - associated with cell 15 from the notebook

   input:
           df: pandas dataframe
           category_lst: list of columns that contain categorical features
           response: string of response name [optional argument that could be used for naming variables or index y column]

   output:
           df: pandas dataframe with new columns for
   '''

    encoded_features = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    for feature in encoded_features:
        df[f'{feature}_Churn'] = df[feature].map(
            df.groupby(feature)['Churn'].mean())
    return df

    df['Card_Category_Churn'] = card_lst

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]

    return X


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def perform_feature_engineering(df, response):
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X = df[keep_cols]
    y = df[response]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    return y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr


def feature_importance_plot(model, output_pth):
    joblib.dump(model, output_pth)


def train_models(X_train, X_test, y_train, y_test):
    y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = classification_report_image(
        X_train, X_test, y_train, y_test)
    feature_importance_plot(y_train_preds_rf, 'model/rf_model.pkl')
    feature_importance_plot(y_test_preds_rf, 'model/lr_model.pkl')
    logging.info(classification_report(y_train, y_train_preds_rf))
    logging.info(classification_report(y_train, y_train_preds_lr))
    logging.info(classification_report(y_test, y_test_preds_rf))
    logging.info(classification_report(y_test, y_test_preds_lr))
