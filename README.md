NCAA March Madness Bracket Predictor

This project uses machine learning to predict the outcomes of NCAA March Madness games. The model is trained on historical tournament and regular season data and uses team level statistics to estimate win probabilities for each matchup.

The goal of the project is to build a clear and reproducible prediction pipeline that mirrors how real world sports analytics systems are designed, trained, and evaluated.

Results

Trained on over 100,000 historical game records spanning multiple seasons

Achieved approximately 78 percent accuracy on held out historical data

Produced calibrated win probabilities rather than binary predictions, making the model suitable for bracket based evaluation and comparison

Modeling Approach

Supervised learning using Logistic Regression and Random Forest models

Feature engineering from team statistics including scoring averages, point margin, tournament seed, and conference strength

Matchup based training examples constructed from differences between opposing team feature vectors

Dataset

The model expects a CSV dataset located at:

data/tournament_data.csv


Each row represents a single game from the regular season or NCAA tournament.

Expected columns

season – Season year of the game

team_id – Unique identifier for the primary team

opp_id – Unique identifier for the opposing team

team_seed – Tournament seed of the primary team

opp_seed – Tournament seed of the opposing team

team_points – Points scored by the primary team

opp_points – Points scored by the opposing team

win – Binary label indicating whether the primary team won

location – Game location (H, A, or N)

game_type – Regular season or tournament game

The data/ directory is intentionally excluded from version control. You may use real Kaggle data or your own historical dataset, as long as it follows the schema above.

How to Run
1. Install dependencies
pip install -r requirements.txt

2. Add the dataset

Create a data/ directory and place tournament_data.csv inside it:

mkdir data
# add tournament_data.csv to this folder

3. Build the training dataset
python data_preprocessing.py

4. Train the model and generate predictions
python predictor.py

Project Structure

predictor.py
Handles model training, evaluation, and matchup level prediction

data_preprocessing.py
Constructs team feature vectors and builds the final training dataset
