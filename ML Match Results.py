from scraping_fbref import scrape_premier_league_fixtures
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
import pandas as pd

# Scrape Premier League fixtures and scores from the 2017/2018 to the 2022/2023 seasons
data = scrape_premier_league_fixtures(2017, 2023)

data.dropna(inplace=True)

# Ensure data is sorted by data
data.sort_values('Date', inplace=True)


# Calculate rolling means for corners
data['Home_Xg_Rolling_Mean'] = data.groupby('Home')['xG'].transform(lambda x: x.shift().rolling(window=20, min_periods=1).mean())
data['Away_Xg_Rolling_Mean'] = data.groupby('Away')['xG.1'].transform(lambda x: x.shift().rolling(window=20, min_periods=1).mean())

print(data.tail())

# Define features and target)

feature_columns = ['Home', 'Away', 'xG', 'xG.1']
X = data[feature_columns]
y = data['Home Goals']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)

print(preds)

print(y_test)