## What is Schrodinger
- Schrodinger is a neural network that is built to predict NBA games
## How does it work
- Schrodinger is trained on 150+ features from histrocial NBA data ranging from 2002-2023
- Schrodinger finds a matchup, retrieves each team's statistics, and uses that to train/predict the outcome of the game 
## Tools used
- Python
- NBA API
- Keras/Tensorflow
- Pandas
- Sklearn
- BeautifulSoup
- gspread
- Odds API
## What I did/learned
- Built a neural network from scratch
- Utilized machine learning techniques such as regularization, feature engineering, feature selection
- Collected data from a variety of different sources
- Cleaned up data and engineered custom features
- Connect to Google Sheets
- 
## Results
- Currently predicts on a ~70% clip
## In Progress
- Organize existing code
- Working on new versions with more data/features
- Data anaylsis on all features
- Modify neural network model structure
## How to use
- Create a virtual environment within the schrodinger directory: python -m venv env
- Activate the virtual environment: source env/bin/activate
- Install the required libraries: pip install -r requirements.txt
- Run main.py
