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
import xgboost as xgb



def scrape_and_clean_data(start_year, end_year):
    """Scrape Premier League fixtures and scores from the 2017/2018 to the 2022/2023 seasons"""
    data = scrape_premier_league_fixtures(start_year, end_year)
    data = data.dropna()
    return data

def initialize_unique_teams(data):
    """Initialize unique teams from the dataset."""
    return pd.concat([data['Home'], data['Away']]).unique()

def expected_result(elo_a, elo_b, elo_width=400):
    """Calculate the expected result of a match between two teams."""
    return 1.0 / (1 + 10 ** ((elo_b - elo_a) / elo_width))

def update_elo_match(winner_elo, loser_elo, draw=False, k_factor=30):
    """Update Elo ratings after a match."""
    expected_win = expected_result(winner_elo, loser_elo)
    if draw:
        change = k_factor * (0.5 - expected_win)
    else:
        change = k_factor * (1 - expected_win)
    winner_elo += change
    loser_elo -= change
    return winner_elo, loser_elo

def update_elos_in_data(data, team_elos, base_elo=1500):
    """Update Elo ratings within the dataset."""
    for index, row in data.iterrows():
        home_team, away_team = row['Home'], row['Away']
        home_elo, away_elo = team_elos.get(home_team, base_elo), team_elos.get(away_team, base_elo)

        data.at[index, 'HomeElo'] = home_elo
        data.at[index, 'AwayElo'] = away_elo

        home_goals, away_goals = row['Home Goals'], row['Away Goals']
        if home_goals > away_goals:
            home_elo, away_elo = update_elo_match(home_elo, away_elo)
        elif home_goals < away_goals:
            away_elo, home_elo = update_elo_match(away_elo, home_elo)
        else:
            home_elo, away_elo = update_elo_match(home_elo, away_elo, draw=True)

        team_elos[home_team] = home_elo
        team_elos[away_team] = away_elo

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
    # Function to add rolling mean features for specified windows and metrics
    def add_rolling_mean_features(data, groupby_col, target_col, windows):
        for window in windows:
            rolling_mean = data.groupby(groupby_col)[target_col].rolling(window=window, min_periods=1, closed='left').mean().reset_index(level=0, drop=True)
            feature_name = f"{groupby_col}_{target_col}_Rolling_Mean_{window}"
            data[feature_name] = rolling_mean

    # Windows for rolling means
    windows = [5,10,20,30]

    # Metrics for which we calculate rolling means
    metrics = ['Home Goals', 'Away Goals', 'xG', 'xG.1']

    # Calculate rolling means for home and away teams
    for metric in metrics:
        add_rolling_mean_features(data, 'Home', metric, windows)
        add_rolling_mean_features(data, 'Away', metric, windows)

    # Calculate rolling mean differences for goals and xG
    for window in windows:
        data[f'Home_Goal_Difference_Rolling_Mean_{window}'] = data[f'Home_Home Goals_Rolling_Mean_{window}'] - data[f'Home_Away Goals_Rolling_Mean_{window}']
        data[f'Away_Goal_Difference_Rolling_Mean_{window}'] = data[f'Away_Away Goals_Rolling_Mean_{window}'] - data[f'Away_Home Goals_Rolling_Mean_{window}']
        
        data[f'Home_Xg_Difference_Rolling_Mean_{window}'] = data[f'Home_xG_Rolling_Mean_{window}'] - data[f'Home_xG.1_Rolling_Mean_{window}']
        data[f'Away_Xg_Difference_Rolling_Mean_{window}'] = data[f'Away_xG_Rolling_Mean_{window}'] - data[f'Away_xG.1_Rolling_Mean_{window}']

        data[f'Home_Minus_Away_Xg_Difference_Rolling_Mean_{window}'] = data[f'Home_Xg_Difference_Rolling_Mean_{window}'] - data[f'Away_Xg_Difference_Rolling_Mean_{window}']
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
    # Define your features and target variable
    rolling_features = [col for col in data.columns if 'Rolling_Mean' in col or 'Difference_Rolling_Mean' in col]
    static_features = ['Year', 'Month_sin', 'Month_cos', 'is_weekend']
    elo_features = ['HomeElo', 'AwayElo']
    feature_cols = static_features + rolling_features + elo_features
    target_col = 'Home Goals'  # Assuming 'Home Goals' is your target variable
    
    # Preprocess Train, Validation, and Test sets
    def preprocess(df):
        # Drop NA values in a way that keeps X and y aligned
        df_filtered = df.dropna(subset=feature_cols + [target_col])
        X = df_filtered[feature_cols]
        y = df_filtered[target_col]
        return X, y

    X_train, y_train = preprocess(train)
    X_validation, y_validation = preprocess(validation)
    X_test, y_test = preprocess(test)

    return X_train, y_train, X_validation, y_validation, X_test, y_test



def train_model(X_train, y_train):
    """Trains a Gradient Boosting Regressor model."""
    model = GradientBoostingRegressor(max_depth=2, min_samples_leaf=1, min_samples_split=2)
    model.fit(X_train, y_train)
    return model

def train_xgb_model(X_train, y_train):
    """Trains an XGBoost Regressor model."""
    model = xgb.XGBRegressor(objective='reg:squarederror', max_depth=3, n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model




































from sklearn.model_selection import BaseCrossValidator
import numpy as np

class BlockedTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]




from sklearn.model_selection import GridSearchCV

def tune_xgb_model(X_train, y_train, param_grid, cv_splits):
    """
    Performs hyperparameter tuning for an XGBoost model using Blocked Cross-Validation.
    
    Parameters:
    - X_train: Training features
    - y_train: Training target variable
    - param_grid: Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
    - cv_splits: Number of splits to use for Blocked Time Series Cross-Validation.
    
    Returns:
    - Best estimator after tuning.
    """
    model = xgb.XGBRegressor(objective='reg:squarederror')
    cv = BlockedTimeSeriesSplit(n_splits=cv_splits)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best CV score:", -grid_search.best_score_)
    
    return grid_search.best_estimator_




from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def tune_xgb_model_with_timeseries_cv(X_train, y_train, param_grid, n_splits):
    """
    Performs hyperparameter tuning for an XGBoost model using Time Series Split Cross-Validation.
    
    Parameters:
    - X_train: Training features
    - y_train: Training target variable
    - param_grid: Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
    - n_splits: Number of splits to use for Time Series Cross-Validation.
    
    Returns:
    - Best estimator after tuning.
    """
    model = xgb.XGBRegressor(objective='reg:squarederror')
    tscv = TimeSeriesSplit(n_splits=n_splits)
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Best parameters found: ", grid_search.best_params_)
    print("Best CV score: ", -grid_search.best_score_)
    
    return grid_search.best_estimator_















def tune_GBR_model(X_train, y_train, param_grid, cv_splits):
    """
    Performs hyperparameter tuning for an XGBoost model using Blocked Cross-Validation.
    
    Parameters:
    - X_train: Training features
    - y_train: Training target variable
    - param_grid: Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
    - cv_splits: Number of splits to use for Blocked Time Series Cross-Validation.
    
    Returns:
    - Best estimator after tuning.
    """
    model = GradientBoostingRegressor()
    cv = BlockedTimeSeriesSplit(n_splits=cv_splits)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best CV score:", -grid_search.best_score_)
    
    return grid_search.best_estimator_


def tune_GBR_model_with_timeseries_cv(X_train, y_train, param_grid, n_splits):
    """
    Performs hyperparameter tuning for an XGBoost model using Time Series Split Cross-Validation.
    
    Parameters:
    - X_train: Training features
    - y_train: Training target variable
    - param_grid: Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
    - n_splits: Number of splits to use for Time Series Cross-Validation.
    
    Returns:
    - Best estimator after tuning.
    """
    model = GradientBoostingRegressor()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Best parameters found: ", grid_search.best_params_)
    print("Best CV score: ", -grid_search.best_score_)
    
    return grid_search.best_estimator_





































def evaluate_model(model, X, y, dataset_name):
    """
    Evaluates the given model on the provided dataset.
    """
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f'Mean Squared Error for {dataset_name} (model and dataset name): {mse:.2f}')
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
    
    print(glm_results.summary())

def evaluate_baseline(y_train, y_validation, y_test):
    """
    Evaluates the baseline model using mean constant predictions.
    """
    mean_Home_Goals_train = y_train.mean()
    mean_Home_Goals_test = y_test.mean()
    print(f'mean home goals is {mean_Home_Goals_test:.2f}')
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
    #print(results_df.tail(50))  # Adjust the number of rows displayed as needed

    print(pd.DataFrame(predictions_test).tail(20))
    sample_team = results_df[results_df['Home'] == 'Brentford']  # Replace 'SampleTeam' with an actual team name
    print("#############################")
    print(sample_team[['Home', 'Away', 'Actual_Home Goals', 'Predicted_Home Goals']].tail(50))





# Set pandas display option
pd.set_option('display.max_columns', None)
#pd.set_option('expand_frame_repr', False)



# Data processing
data = scrape_and_clean_data(2017, 2024)
unique_teams = initialize_unique_teams(data)
team_elos = {team: 1500 for team in unique_teams}
data = update_elos_in_data(data, team_elos)
data = add_total_goals_feature(data)
data = convert_date_and_add_features(data)
data = calculate_rolling_means(data)
data = create_and_concat_dummy_variables(data)

print(data.sample(12))

# Split data and apply encoding
train, validation, test = split_data(data)
apply_target_encoding(train, validation, test)

# Feature preparation and model training
X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_feature_columns(data, train, validation, test)
model = train_model(X_train, y_train)
xgb_model = train_xgb_model(X_train, y_train)




param_grid = {
    'max_depth': [2, 3, 4, 5],
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
}


# param_grid = {
#     'max_depth': [2],
#     'n_estimators': [50],
#     'learning_rate': [0.05]
# }

# Adjust the number of splits based on your dataset size and temporal granularity
cv_splits = 5








# List of tuning functions for each model
tuning_functions = [
    # (tune_xgb_model, 'XGB Tuned'),
    # (tune_xgb_model_with_timeseries_cv, 'XGB TS Tuned'),
    (tune_GBR_model, 'GBR Tuned'),
    # (tune_GBR_model_with_timeseries_cv, 'GBR TS Tuned')
]

# Iterate over each tuning function and evaluate the tuned models
for tune_func, model_name in tuning_functions:
    # Tuning the model
    best_model = tune_func(X_train, y_train, param_grid, cv_splits)
    
    # Evaluating the tuned model on validation and test sets
    evaluate_model(best_model, X_validation, y_validation, model_name + ' Validation')
    evaluate_model(best_model, X_test, y_test, model_name + ' Test')


















# Model predictions and evaluations
predictions_validation = model.predict(X_validation)
predictions_test = model.predict(X_test)

#for model in [model, xgb_model, best_xgb_model_blocked_ts_cv, best_xgb_model_ts_cv, best_GBR_model_blocked_ts_cv, best_GBR_model_ts_cv]:
for model in [model, xgb_model]:
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















