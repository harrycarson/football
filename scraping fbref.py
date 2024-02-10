import pandas as pd

# Initialize an empty DataFrame to store all the data
all_seasons_df = pd.DataFrame()

# Loop through each season from 2017 to 2023
for season_start_year in range(2017, 2024):
    season_end_year = season_start_year + 1
    url = f'https://fbref.com/en/comps/9/{season_start_year}-{season_end_year}/schedule/{season_start_year}-{season_end_year}-Premier-League-Scores-and-Fixtures'
    
    # Read the HTML tables from the current season's page
    tables = pd.read_html(url)
    
    # Assuming the first table contains the fixtures and scores, append it to the all_seasons_df DataFrame
    all_seasons_df = pd.concat([all_seasons_df, tables[0]], ignore_index=True)

# Define a function to extract home and away goals, handling the en dash
def extract_goals(score):
    if pd.isna(score):
        return None, None  # Handle NaN values right away
    score = score.strip()  # Remove any leading/trailing whitespace
    if '–' in score:  # Use the en dash character for splitting
        home_goals, away_goals = score.split('–', 1)  # Split on the first en dash
        return home_goals.strip(), away_goals.strip()  # Strip any extra spaces after splitting
    else:
        return None, None  # Return None if the format is not as expected

# Apply the function to extract goals
extracted_goals = all_seasons_df['Score'].apply(lambda score: extract_goals(score))
all_seasons_df['Home Goals'] = extracted_goals.apply(lambda x: x[0] if x is not None else None)
all_seasons_df['Away Goals'] = extracted_goals.apply(lambda x: x[1] if x is not None else None)

# Convert 'Home Goals' and 'Away Goals' to numeric, safely handling errors
all_seasons_df['Home Goals'] = pd.to_numeric(all_seasons_df['Home Goals'], errors='coerce')
all_seasons_df['Away Goals'] = pd.to_numeric(all_seasons_df['Away Goals'], errors='coerce')

# Optionally, you might want to drop rows where 'Home Goals' or 'Away Goals' are NaN if those rows aren't useful for your analysis
# all_seasons_df.dropna(subset=['Home Goals', 'Away Goals'], inplace=True)

# Display the first few rows to verify the corrections
print(all_seasons_df[['Day', 'Date', 'Time', 'Home', 'Away', 'Home Goals', 'Away Goals', 'xG', 'xG.1']].tail())
