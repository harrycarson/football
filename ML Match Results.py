from scraping_fbref import scrape_premier_league_fixtures

# Scrape Premier League fixtures and scores from the 2017/2018 to the 2022/2023 seasons
df = scrape_premier_league_fixtures(2017, 2023)
print(df.tail())