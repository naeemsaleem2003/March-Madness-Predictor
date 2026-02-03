# NCAA March Madness Bracket Predictor

This project uses machine learning to predict the outcomes of NCAA March Madness games. The model is trained on historical regular season and tournament data and uses team level statistics to estimate win probabilities for each matchup.

## Features
- Supervised learning using Logistic Regression and Random Forest models
- Feature engineering from team statistics such as scoring averages, point margin, tournament seed, and conference strength
- Probability based predictions suitable for bracket style evaluation
- Achieves approximately 78% accuracy on held out historical data

## How to Run
1. Install dependencies:
2. Add the dataset:
- Create a `data/` directory
- Place `tournament_data.csv` inside the folder
3. Build the training dataset:
4. Train the model and generate predictions:

## Dataset
The model expects a CSV file located at `data/tournament_data.csv`.  
Each row represents a single game and includes team identifiers, seeds, scores, and the game outcome.

## Files
- `predictor.py`: Main model training, evaluation, and matchup level prediction logic
- `data_preprocessing.py`: Builds team feature vectors and constructs the final training dataset
- `data/`: Contains the historical dataset used for training (not committed to the repository)
