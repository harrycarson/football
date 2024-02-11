import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
import graphviz
import statsmodels.api as sm

# Set the option to None to display all columns
pd.set_option('display.max_columns', None)

# More data!
data_23_24 = pd.read_csv("C:/Users/harry/Documents/Actual Documents/Football-Data/EPL 23-24 Dataset Football-Data.csv")
data_22_23 = pd.read_csv("C:/Users/harry/Documents/Actual Documents/Football-Data/EPL 22-23 Dataset Football-Data.csv")
data_21_22 = pd.read_csv("C:/Users/harry/Documents/Actual Documents/Football-Data/EPL 21-22 Dataset Football-Data.csv")
data_20_21 = pd.read_csv("C:/Users/harry/Documents/Actual Documents/Football-Data/EPL 20-21 Dataset Football-Data.csv")
data_19_20 = pd.read_csv("C:/Users/harry/Documents/Actual Documents/Football-Data/EPL 19-20 Dataset Football-Data.csv")
data_18_19 = pd.read_csv("C:/Users/harry/Documents/Actual Documents/Football-Data/EPL 18-19 Dataset Football-Data.csv")
data_17_18 = pd.read_csv("C:/Users/harry/Documents/Actual Documents/Football-Data/EPL 17-18 Dataset Football-Data.csv")
data_16_17 = pd.read_csv("C:/Users/harry/Documents/Actual Documents/Football-Data/EPL 16-17 Dataset Football-Data.csv")
data_15_16 = pd.read_csv("C:/Users/harry/Documents/Actual Documents/Football-Data/EPL 15-16 Dataset Football-Data.csv")
data_14_15 = pd.read_csv("C:/Users/harry/Documents/Actual Documents/Football-Data/EPL 14-15 Dataset Football-Data.csv")
data_13_14 = pd.read_csv("C:/Users/harry/Documents/Actual Documents/Football-Data/EPL 13-14 Dataset Football-Data.csv")

# Combine data
data = pd.concat([data_22_23], ignore_index=True)
print(data.tail())
split_point = int(len(data) * 1)
data = data.iloc[:split_point]

data['FTTG'] = data['FTHG'] + data['FTAG']

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

print(data.head())


# Add additional features
data['Year'] = data['Date'].dt.year
data['is_weekend'] = (data['Date'].dt.dayofweek >= 5).astype(int)
data['Month_sin'] = np.sin(data['Date'].dt.month * 2 * np.pi / 12)
data['Month_cos'] = np.cos(data['Date'].dt.month * 2 * np.pi / 12)

# After adding the new features
print("Shape after adding new features:", data.shape)
print(data.tail())

# Calculate rolling means
rolling_means_home_scored = data.groupby('HomeTeam')['FTHG'].rolling(window=10, min_periods=1, closed='left').mean().reset_index(level=0, drop=True)
rolling_means_home_conceded = data.groupby('HomeTeam')['FTAG'].rolling(window=10, min_periods=1, closed='left').mean().reset_index(level=0, drop=True)
rolling_means_away_conceded = data.groupby('AwayTeam')['FTHG'].rolling(window=10, min_periods=1, closed='left').mean().reset_index(level=0, drop=True)
rolling_means_away_scored = data.groupby('AwayTeam')['FTAG'].rolling(window=10, min_periods=1, closed='left').mean().reset_index(level=0, drop=True)


data['Home_Scored_Rolling_Mean'] = rolling_means_home_scored
data['Home_Conceded_Rolling_Mean'] = rolling_means_home_conceded
data['Away_Scored_Rolling_Mean'] = rolling_means_away_scored
data['Away_Conceded_Rolling_Mean'] = rolling_means_away_conceded



# Goal difference
data['Home_Goal_Difference_Rolling_Mean'] = data['Home_Scored_Rolling_Mean'] - data['Home_Conceded_Rolling_Mean']
data['Away_Goal_Difference_Rolling_Mean'] = data['Away_Scored_Rolling_Mean'] - data['Away_Conceded_Rolling_Mean']

# After calculating rolling means
print("Shape after calculating rolling means:", data.shape)
print(data.head())

# After dropping NaN values
print("Shape after dropping NaN values:", data.shape)
print(data.tail())

# Create dummy variables
dummy_away = pd.get_dummies(data['AwayTeam'], prefix='AwayTeam_')
dummy_home = pd.get_dummies(data['HomeTeam'], prefix='HomeTeam_')

# Concatenate the dummy variables with the original DataFrame
datax = pd.concat([data, dummy_away, dummy_home], axis=1)

print(data)

# Split the data
train_size = int(len(data) * 0.6)
validation_size = int(len(data) * 0.2)
train = data.iloc[:train_size]
validation = data.iloc[train_size:train_size + validation_size]
test = data.iloc[train_size + validation_size:]

# Function for mean target encoding
def target_encode_simple(df_train, df, col, target):
    target_mean = df_train.groupby(col)[target].mean()
    return df[col].map(target_mean).fillna(df_train[target].mean())

# Apply target encoding
for col in ['HomeTeam', 'AwayTeam']:
    train.loc[:, col + '_target_mean'] = target_encode_simple(train, train, col, 'FTHG')
    validation.loc[:, col + '_target_mean'] = target_encode_simple(train, validation, col, 'FTHG')
    test.loc[:, col + '_target_mean'] = target_encode_simple(train, test, col, 'FTHG')

#####
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize_scalar

def find_lambda(odds_over_2_5_goals):

    prob_over_2_5_goals = 1/odds_over_2_5_goals

    # Adjust the bounds if necessary, based on your data distribution
    bounds = (0, 10)  
    
    def objective(lmbda):
        return abs((1 - poisson.cdf(2, lmbda)) - prob_over_2_5_goals)

    result = minimize_scalar(objective, bounds=bounds, method='bounded')
    
    if result.success:
        return result.x
    else:
        # Handle edge cases or return a fallback value
        if prob_over_2_5_goals > 0.99:
            return bounds[1]  # High lambda for high probabilities
        elif prob_over_2_5_goals < 0.01:
            return bounds[0]  # Low lambda for low probabilities
        return np.nan  # Or return NaN or a default value

# Apply the function to the 'B365>2.5' column
data['Mean_Goals'] = data['B365>2.5'].apply(find_lambda)

# First, combine all unique columns from both dataframes
combined_columns = set(data.columns) | set(train.columns)

# Then, filter columns based on your criteria
all_cols = [
    col for col in combined_columns
    if col.startswith('AwayTeam') 
    or col.startswith('HomeTeam') 
    or col.endswith('target_mean') 
    or col in [
        'Year', 'Month_sin', 'Month_cos', 
        'Home_Scored_Rolling_Mean', 'Home_Conceded_Rolling_Mean', 
        'Away_Scored_Rolling_Mean', 'Away_Conceded_Rolling_Mean',
        'Home_Goal_Difference_Rolling_Mean', 'Away_Goal_Difference_Rolling_Mean', 
        'FTTG'
    ]
]

print(f'these are all the cols {all_cols}')


feature_cols = [
    col for col in all_cols 
    if col.startswith('AwayTeam_') 
    or col.startswith('HomeTeam_')
    or col.endswith('target_mean') 
    or col in [
        'Year'
        ,'Month_sin'
        ,'Month_cos'
        ,'Home_Scored_Rolling_Mean' 
        ,'Home_Conceded_Rolling_Mean' 
        ,'Away_Scored_Rolling_Mean'
        ,'Away_Conceded_Rolling_Mean'
        ,'Home_Goal_Difference_Rolling_Mean'
        ,'Away_Goal_Difference_Rolling_Mean'
    ]
]

print(feature_cols)
print(train)

train1 = train[all_cols].dropna()
validation1 = validation[all_cols].dropna()
test1 = test[all_cols].dropna()

X_train = train1[feature_cols]
y_train = train1['FTTG']
X_validation = validation1[feature_cols]
y_validation = validation1['FTTG']
X_test = test1[feature_cols]
y_test = test1['FTTG']


# After dropping NaN values
print("Shape after dropping NaN values:", data.shape)
print(data.tail())


# Before fitting the model
print("Shape of X_train:", X_train.shape)
print("X_train head:", X_train.head())


# Train the model
model = GradientBoostingRegressor(max_depth=2, min_samples_leaf=1, min_samples_split=2)
model.fit(X_train, y_train)

# Evaluation on validation and test sets
for X, y, dataset_name in [(X_validation, y_validation, 'Validation'), (X_test, y_test, 'Test')]:
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f'Mean Squared Error on {dataset_name} Set: {mse}')




# Calculate the mean of the target variable in the training set
mean_fthg_train = y_train.mean()

# Create constant predictions for the validation and test sets
constant_predictions_validation = np.full_like(y_validation, mean_fthg_train)
constant_predictions_test = np.full_like(y_test, mean_fthg_train)

# Calculate MSE for constant predictions
mse_constant_validation = mean_squared_error(y_validation, constant_predictions_validation)
mse_constant_test = mean_squared_error(y_test, constant_predictions_test)

# Print the MSE for constant predictions
print(f'Mean Squared Error with Constant Predictions on Validation Set: {mse_constant_validation}')
print(f'Mean Squared Error with Constant Predictions on Test Set: {mse_constant_test}')

# Feature importance and visualization
feature_importances = model.feature_importances_
features = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

# Display feature importance
print(feature_importance_df)

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in Gradient Boosting Model')
plt.gca().invert_yaxis()
plt.show()

# Model Evaluation
predictions_test = model.predict(X_test)
mse_test = mean_squared_error(y_test, predictions_test)
print(f'Mean Squared Error on Test Set: {mse_test}')

# Visualization of Actual vs Predicted Goals
plt.scatter(y_test, predictions_test)
plt.xlabel('Actual Goals')
plt.ylabel('Predicted Goals')
plt.title('Actual vs Predicted Goals on Test Set')
plt.show()

print(y_test[-5:], predictions[-5:])

# Assuming 'y_test' contains the actual goals and 'predictions_test' contains the predicted goals

# Convert predictions to a DataFrame
predicted_goals_df = pd.DataFrame(predictions_test, columns=['Predicted_FTHG'])

# Reset index of y_test to align with the index of predicted_goals_df
y_test_reset = y_test.reset_index(drop=True)

# Combine the test set information with the predictions
results_df = pd.concat([test[['Date', 'HomeTeam', 'AwayTeam']].reset_index(drop=True), y_test_reset, predicted_goals_df], axis=1)

# Rename columns for clarity
results_df.rename(columns={'FTHG': 'Actual_FTHG'}, inplace=True)

# Display the DataFrame
print(results_df.tail(10))  # Adjust the number of rows displayed as needed

# Evaluation on validation and test sets
for X, y, dataset_name in [(X_validation, y_validation, 'Validation'), (X_test, y_test, 'Test')]:
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f'Mean Squared Error on {dataset_name} Set: {mse}')

# Calculate the mean of the target variable in the training set
mean_fthg_train = y_train.mean()

# Create constant predictions for the validation and test sets
constant_predictions_validation = np.full_like(y_validation, mean_fthg_train)
constant_predictions_test = np.full_like(y_test, mean_fthg_train)

#bet365 = mean_squared_error(y_test, data['Mean_Goals'])
#print(bet365)
#print(f'Mean Squared Error bet365 over 2.5 on Validation Set: {bet365}')


# Calculate MSE for constant predictions
mse_constant_validation = mean_squared_error(y_validation, constant_predictions_validation)
mse_constant_test = mean_squared_error(y_test, constant_predictions_test)

# Print the MSE for constant predictions
print(f'Mean Squared Error with Constant Predictions on Validation Set: {mse_constant_validation}')
print(f'Mean Squared Error with Constant Predictions on Test Set: {mse_constant_test}')


# Add constant to training, validation, and test datasets
X_train_glm = sm.add_constant(X_train, prepend=False, has_constant='add')
X_validation_glm = sm.add_constant(X_validation, prepend=False, has_constant='add')
X_test_glm = sm.add_constant(X_test, prepend=False, has_constant='add')

print(X_test_glm)

# Fit the GLM
glm_family = sm.families.Poisson()
glm_model = sm.GLM(y_train, X_train_glm, family=glm_family)
glm_results = glm_model.fit()

# Print GLM summary
print(glm_results.summary())


print(X_test_glm)


# GLM Predictions and evaluation
for X, y, dataset_name in [(X_validation_glm, y_validation, 'Validation'), (X_test_glm, y_test, 'Test')]:
    predictions_glm = glm_results.predict(X)
    mse_glm = mean_squared_error(y, predictions_glm)
    print(f'Mean Squared Error for GLM on {dataset_name} Set: {mse_glm}')