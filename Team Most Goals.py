import numpy as np
from scipy.stats import poisson

def odds_to_implied_probabilities(home_odds, draw_odds, away_odds):
    """
    Convert decimal odds to implied probabilities and adjust for overround.
    """
    home_prob = 1 / home_odds
    draw_prob = 1 / draw_odds
    away_prob = 1 / away_odds
    
    total_market = home_prob + draw_prob + away_prob
    # Adjust probabilities to account for the total market (overround)
    home_prob_adjusted = home_prob / total_market
    draw_prob_adjusted = draw_prob / total_market
    away_prob_adjusted = away_prob / total_market
    
    return home_prob_adjusted, draw_prob_adjusted, away_prob_adjusted

def estimate_xg_from_probabilities(home_prob, away_prob):
    """
    Estimate xG values for home and away teams based on adjusted probabilities.
    This function assumes a simplification for converting probabilities to xG.
    """
    # Simplistic estimation of xG from probabilities - more complex models would be required for accurate conversion
    # These estimations are placeholders and should be replaced with a model based on historical data
    home_xg = -np.log(1 - home_prob)
    away_xg = -np.log(1 - away_prob)
    
    return home_xg, away_xg

def calculate_xg(home_odds, draw_odds, away_odds):
    """
    Calculate xG values for home and away teams based on their odds including draw odds.
    """
    home_prob, draw_prob, away_prob = odds_to_implied_probabilities(home_odds, draw_odds, away_odds)
    # Estimate xG based on the probability of winning (not accounting for draws directly)
    home_xg, away_xg = estimate_xg_from_probabilities(home_prob, away_prob)
    
    return home_xg, away_xg

# Example usage
home_odds, draw_odds, away_odds = 2.5, 3.2, 2.8  # Example odds
home_xg, away_xg = calculate_xg(home_odds, draw_odds, away_odds)    
print(f"Estimated xG: Home Team = {home_xg:.2f}, Away Team = {away_xg:.2f}")

