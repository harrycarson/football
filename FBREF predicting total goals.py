from scraping_fbref import scrape_premier_league_fixtures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
import graphviz
import statsmodels.api as sm
from sklearn.feature_selection import RFE


def scrape_and_clean_data(start_year, end_year):
    """Scrape Premier League fixtures and scores from the 2017/2018 to the 2022/2023 seasons"""
    data = scrape_premier_league_fixtures(start_year, end_year)
    data = data.dropna()
    return data

def add_total_goals_feature(data):
    data['FTTG'] = data['Home Goals'] + data['Away Goals']
    return data

def convert_date_and_add_features(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['is_weekend'] = (data['Date'].dt.dayofweek >= 5).astype(int)
    data['Month_sin'] = np.sin(data['Date'].dt.month * 2 * np.pi / 12)
    data['Month_cos'] = np.cos(data['Date'].dt.month * 2 * np.pi / 12)
    return data

def calculate_rolling_means(data):

    # Calculate rolling means
    rolling_means_home_scored = data.groupby('Home')['Home Goals'].rolling(window=10, min_periods=1, closed='left').mean().reset_index(level=0, drop=True)
    rolling_means_home_conceded = data.groupby('Home')['Away Goals'].rolling(window=10, min_periods=1, closed='left').mean().reset_index(level=0, drop=True)
    rolling_means_away_conceded = data.groupby('Away')['Home Goals'].rolling(window=10, min_periods=1, closed='left').mean().reset_index(level=0, drop=True)
    rolling_means_away_scored = data.groupby('Away')['Away Goals'].rolling(window=10, min_periods=1, closed='left').mean().reset_index(level=0, drop=True)

    rolling_means_home_xg = data.groupby('Home')['xG'].rolling(window=40, min_periods=1, closed='left').mean().reset_index(level=0, drop=True)
    rolling_means_away_xg = data.groupby('Home')['xG.1'].rolling(window=40, min_periods=1, closed='left').mean().reset_index(level=0, drop=True)

    data['Home_Scored_Rolling_Mean'] = rolling_means_home_scored
    data['Home_Conceded_Rolling_Mean'] = rolling_means_home_conceded
    data['Away_Scored_Rolling_Mean'] = rolling_means_away_scored
    data['Away_Conceded_Rolling_Mean'] = rolling_means_away_conceded

    data['Home_Xg_Rolling_Mean'] = rolling_means_home_xg
    data['Away_Xg_Rolling_Mean'] = rolling_means_away_xg

    # Goal difference
    data['Home_Goal_Difference_Rolling_Mean'] = data['Home_Scored_Rolling_Mean'] - data['Home_Conceded_Rolling_Mean']
    data['Away_Goal_Difference_Rolling_Mean'] = data['Away_Scored_Rolling_Mean'] - data['Away_Conceded_Rolling_Mean']

    return data

def create_and_concat_dummy_variables(data):
    dummy_away = pd.get_dummies(data['Away'], prefix='Away')
    dummy_home = pd.get_dummies(data['Home'], prefix='Home')
    data = pd.concat([data, dummy_away, dummy_home], axis=1)
    return data

def split_data(data):
    train_size = int(len(data) * 0.6)
    validation_size = int(len(data) * 0.2)
    train = data.iloc[:train_size]
    validation = data.iloc[train_size:train_size + validation_size]
    test = data.iloc[train_size + validation_size:]
    return train, validation, test

def target_encode_simple(df_train, df, col, target):
    target_mean = df_train.groupby(col)[target].mean()
    return df[col].map(target_mean).fillna(df_train[target].mean())

def apply_target_encoding(train, validation, test):
    for col in ['Home', 'Away']:
        for df in [train, validation, test]:
            target_encode_simple(train, df, col, 'Home Goals')

def prepare_feature_columns(data, train, validation, test):
    """
    Prepares the feature columns based on specified criteria and splits the data into features (X) and target (y) sets.
    """
    combined_columns = set(data.columns) | set(train.columns)

    all_cols = [
        col for col in combined_columns
        if col.startswith('Away') or col.startswith('Home') or col.endswith('target_mean') or col in [
            'Year', 'Month_sin', 'Month_cos', 
            'Home_Scored_Rolling_Mean', 'Home_Conceded_Rolling_Mean', 
            'Away_Scored_Rolling_Mean', 'Away_Conceded_Rolling_Mean',
            'Home_Goal_Difference_Rolling_Mean', 'Away_Goal_Difference_Rolling_Mean', 
            'FTTG', 'Home_Xg_Rolling_Mean', 'Away_Xg_Rolling_Mean'
        ]
    ]

    feature_cols = [
        col for col in all_cols 
        if col in [
            'Year', 'Month_sin', 'Month_cos', 
            'Home_Scored_Rolling_Mean', 'Home_Conceded_Rolling_Mean', 
            'Away_Scored_Rolling_Mean', 'Away_Conceded_Rolling_Mean',
            'Home_Goal_Difference_Rolling_Mean', 'Away_Goal_Difference_Rolling_Mean', 
            'Home_Xg_Rolling_Mean', 'Away_Xg_Rolling_Mean'
        ]
    ]

    train_prepared = train[all_cols].dropna()
    validation_prepared = validation[all_cols].dropna()
    test_prepared = test[all_cols].dropna()

    X_train = train_prepared[feature_cols]
    y_train = train_prepared['Home Goals']
    X_validation = validation_prepared[feature_cols]
    y_validation = validation_prepared['Home Goals']
    X_test = test_prepared[feature_cols]
    y_test = test_prepared['Home Goals']

    return X_train, y_train, X_validation, y_validation, X_test, y_test

def train_model(X_train, y_train):
    """Trains a Gradient Boosting Regressor model."""
    model = GradientBoostingRegressor(max_depth=2, min_samples_leaf=1, min_samples_split=2)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y, dataset_name):
    """
    Evaluates the given model on the provided dataset.
    """
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f'Mean Squared Error on {dataset_name} Set: {mse:.2f}')
    return mse

def fit_and_evaluate_glm(y_train, X_train, y_validation, X_validation, y_test, X_test):
    X_train_glm = sm.add_constant(X_train, prepend=False, has_constant='add')
    X_validation_glm = sm.add_constant(X_validation, prepend=False, has_constant='add')
    X_test_glm = sm.add_constant(X_test, prepend=False, has_constant='add')
    
    glm_family = sm.families.Poisson()
    glm_model = sm.GLM(y_train, X_train_glm, family=glm_family)
    glm_results = glm_model.fit()
    
    for X, y, dataset_name in [(X_validation_glm, y_validation, 'Validation'), (X_test_glm, y_test, 'Test')]:
        predictions_glm = glm_results.predict(X)
        mse_glm = mean_squared_error(y, predictions_glm)
        print(f'Mean Squared Error for GLM on {dataset_name} Set: {mse_glm:.2f}')
    
    #print(glm_results.summary())

def evaluate_baseline(y_train, y_validation, y_test):
    """
    Evaluates the baseline model using mean constant predictions.
    """
    mean_Home_Goals_train = y_train.mean()
    mse_constant_validation = mean_squared_error(y_validation, np.full_like(y_validation, mean_Home_Goals_train))
    mse_constant_test = mean_squared_error(y_test, np.full_like(y_test, mean_Home_Goals_train))
    
    print(f'Mean Squared Error with Constant Predictions on Validation Set: {mse_constant_validation:.2f}')
    print(f'Mean Squared Error with Constant Predictions on Test Set: {mse_constant_test:.2f}')

def visualize_feature_importances(model, X_train):
    feature_importances = model.feature_importances_
    features = X_train.columns
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances in Gradient Boosting Model')
    plt.gca().invert_yaxis()
    plt.show()

def visualize_actual_vs_predicted(y_test, predictions_test):
    plt.scatter(y_test, predictions_test)
    plt.xlabel('Actual Goals')
    plt.ylabel('Predicted Goals')
    plt.title('Actual vs Predicted Goals on Test Set')
    plt.show()

def combine_predictions_with_actual(test, X_test, y_test, predictions_test):
    """
    Combines the test set with actual values, predictions, and the features used for predictions.
    """
    predicted_goals_df = pd.DataFrame(predictions_test, columns=['Predicted_Home Goals'])
    y_test_reset = y_test.reset_index(drop=True)
    X_test_reset = X_test.reset_index(drop=True)
    results_df = pd.concat([test[['Date', 'Home', 'Away']].reset_index(drop=True), X_test_reset, y_test_reset, predicted_goals_df], axis=1)
    results_df.rename(columns={'Home Goals': 'Actual_Home Goals'}, inplace=True)
    print(results_df.tail(50))  # Adjust the number of rows displayed as needed





# Set pandas display option
pd.set_option('display.max_columns', None)
#pd.set_option('expand_frame_repr', False)



# Data processing
data = scrape_and_clean_data(2017, 2023)
data = add_total_goals_feature(data)
data = convert_date_and_add_features(data)
data = calculate_rolling_means(data)
data = create_and_concat_dummy_variables(data)

# Split data and apply encoding
train, validation, test = split_data(data)
apply_target_encoding(train, validation, test)

# Feature preparation and model training
X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_feature_columns(data, train, validation, test)
model = train_model(X_train, y_train)

# Model predictions and evaluations
predictions_validation = model.predict(X_validation)
predictions_test = model.predict(X_test)
evaluate_model(model, X_validation, y_validation, 'Validation')
evaluate_model(model, X_test, y_test, 'Test')
fit_and_evaluate_glm(y_train, X_train, y_validation, X_validation, y_test, X_test)

# Baseline evaluation
evaluate_baseline(y_train, y_validation, y_test)

# Visualization and additional model evaluation
visualize_feature_importances(model, X_train)
visualize_actual_vs_predicted(y_test, predictions_test)
#combine_predictions_with_actual(data, test, y_test, predictions_test)

X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_feature_columns(data, train, validation, test)
combine_predictions_with_actual(test, X_test, y_test, predictions_test)





