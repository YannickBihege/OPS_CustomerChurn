import os
import pandas as pd
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    """
	test data import - this example is completed for you to assist with the other test functions
	"""
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, test_df):
    """
	test perform eda function
	"""
    try:
        result = perform_eda(test_df)
        assert result is None
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: The function should return None")
        raise err


def test_encoder_helper(encoder_helper, category_lst, response, test_df):
    '''
	test encoder helper
	'''
    try:
        result = encoder_helper(df, category_lst, response)
        assert isinstance(result, pd.DataFrame)
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: The output should be a DataFrame")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, response, test_df):
    '''
	test perform_feature_engineering
	'''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, response)
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: Output shapes are not appropriate")
        raise err


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
	test train_models
	'''
    try:
        result = train_models(X_train, X_test, y_train, y_test)
        assert result is None
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models: The function should return None")
        raise err


if __name__ == "__main__":
    # Used a random small dataframe (test_df) for testing purposes.
    category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    response = 'Response'
    df = pd.DataFrame({
        'Gender': ['M', 'F', 'M', 'M', 'F'],
        'Education_Level': ['Grad', 'Grad', 'Undergrad', 'Grad', 'Undergrad'],
        'Marital_Status': ['Single', 'Married', 'Single', 'Single', 'Married'],
        'Income_Category': ['60K', '70K', '60K', '80K', '70K'],
        'Card_Category': ['Blue', 'Silver', 'Blue', 'Gold', 'Silver'],
        'Response': [0, 1, 0, 0, 1]
    })

    test_import(cls.import_data)
    test_eda(cls.perform_eda, df)
    test_encoder_helper(cls.encoder_helper, category_lst, response, df)
    test_perform_feature_engineering(cls.perform_feature_engineering, response, df)
    test_train_models(cls.train_models, X_train, X_test, y_train, y_test)
