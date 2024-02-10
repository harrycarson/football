import pandas as pd

def scrape_premier_league_fixtures(start_year, end_year):
    """
    Scrape Premier League fixtures and scores from fbref.com for a range of seasons.

    Parameters:
    - start_year: int, the starting season year (e.g., 2017 for the 2017/2018 season)
    - end_year: int, the ending season year (inclusive)

    Returns:
    - A pandas DataFrame containing the scraped data.
    """
    all_seasons_df = pd.DataFrame()

    for season_start_year in range(start_year, end_year + 1):
        season_end_year = season_start_year + 1
        url = f'https://fbref.com/en/comps/9/{season_start_year}-{season_end_year}/schedule/{season_start_year}-{season_end_year}-Premier-League-Scores-and-Fixtures'
        tables = pd.read_html(url)
        all_seasons_df = pd.concat([all_seasons_df, tables[0]], ignore_index=True)

    def extract_goals(score):
        if pd.isna(score):
            return None, None
        score = score.strip()
        if '–' in score:
            home_goals, away_goals = score.split('–', 1)
            return home_goals.strip(), away_goals.strip()
        else:
            return None, None

    extracted_goals = all_seasons_df['Score'].apply(lambda score: extract_goals(score))
    all_seasons_df['Home Goals'] = extracted_goals.apply(lambda x: x[0] if x is not None else None)
    all_seasons_df['Away Goals'] = extracted_goals.apply(lambda x: x[1] if x is not None else None)

    all_seasons_df['Home Goals'] = pd.to_numeric(all_seasons_df['Home Goals'], errors='coerce')
    all_seasons_df['Away Goals'] = pd.to_numeric(all_seasons_df['Away Goals'], errors='coerce')

    return all_seasons_df[['Day', 'Date', 'Time', 'Home', 'Away', 'Home Goals', 'Away Goals', 'xG', 'xG.1']]

# Example usage:
# Scrape Premier League fixtures and scores from the 2017/2018 to the 2022/2023 seasons
df = scrape_premier_league_fixtures(2017, 2023)
print(df.tail())
