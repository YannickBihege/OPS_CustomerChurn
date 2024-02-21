# Churn Library

This library is a set of utility modules used for data preprocessing and customer churn analysis. 

## Dependencies

This project requires Python and the following Python libraries installed:

- os
- pandas
- numpy
- seaborn 
- sklearn
- joblib
- matplotlib
- logging

If Python is not yet installed on your machine, we recommend using [Anaconda](https://www.anaconda.com/download/), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

## Modules

### `import_data(pth)`

Returns a dataframe for the `csv` file found at `pth`.

**Inputs:**
- `pth`: a path to the `csv`

**Output:**
- A pandas dataframe

### `perform_eda(df)`

Performs exploratory data analysis on the dataframe and saves plots to disk.

**Inputs:**
- `df`: Input dataframe for EDA.

**Output:** None (but outputs plots to disk)

### `encoder_helper(df, category_lst, response)`

Transforms each categorical column into a new column for each category, indicating the proportion of churn.

**Inputs:**
- `df`: pandas dataframe
- `category_lst`: list of columns that contain categorical features
- `response`: string of response name 

**Output:**
- pandas dataframe with new columns for

### `perform_feature_engineering(df, response)`

Splits the data into training and test data.

**Inputs:**
- `df`: pandas dataframe 
- `response`: string of response name

**Output:**
- X_train, X_test, y_train, y_test : Training and testing data

### `classification_report_image(X_train, X_test, y_train, y_test)`

Classifies the data using a random forest classifier and a logistic regression model, and predicts the test data.

**Inputs:**
- X_train, X_test, y_train, y_test : Training and testing data

**Output:**
- y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr : Predicted results

### `feature_importance_plot(model, output_pth)`

Saves the fitted model at `output_pth`.

**Inputs:**
- `model`: the fitted model
- `output_pth`: path to save the model

### `train_models(X_train, X_test, y_train, y_test)`

Trains and saves two models.

**Inputs:**
- X_train, X_test, y_train, y_test : Training and testing data

**Output:**
- None

## Testing

Contains several test functions to make sure all the functions are working properly.
Generally, a small dataframe is created for testing purposes ("test_df" in the code). 
A list of category column names ("category_lst") is declared independently, and then, 
the response column name ("response") is specified.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.