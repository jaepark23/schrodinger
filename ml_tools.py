import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class MLTools:
    def load_training_data(directory_path : str) -> pd.DataFrame:
        """
        Iterates through directory of training data in csv format and returns DataFrame of all combined. Along with some simple
        preprocessing of data. 

        directory_path (str) : directory to iterate and merge files from

        Returns:
        (pd.DataFrame) : DataFrame of all data combined 
        """
        directory = os.fsencode(directory_path)
        training_data = []
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".csv"):
                csv_path = directory_path + filename
                try:
                    training_data.append(pd.read_csv(csv_path, index_col=0).loc[:, ~pd.read_csv(csv_path, index_col=0).columns.duplicated()])
                except Exception as e:
                    print(e)
        training_data = pd.concat(training_data)

        return training_data

    def preprocess_training_data(training_data : pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses training data by dropping uninterested columns and cleaning up discrepencies. 

        training_data (pd.DataFrame) : data to preprocess 

        Returns:
        (pd.DataFrame) : cleaned data
        """
        # remove duplicate columns upon concatenating
        for col in training_data.columns:
            if ".1" in col:
                training_data.drop(columns=[col], inplace=True) 
        # drop uneccessary columns
        training_data.drop(columns=["HOME_TEAM_NAME", "AWAY_TEAM_NAME", "DATE"], inplace=True)
        training_data.dropna(inplace=True)
        # drop columns that have RANK in them (rank columns)
        columns_to_drop = training_data.filter(like="RANK").columns.tolist()
        columns_to_drop.append("W_PCT") 
        training_data.drop(columns=columns_to_drop, inplace=True)
        return training_data

    def prep_data(training_data : pd.DataFrame, scaler_path : str = None) -> tuple[list, list]:
        """
        Splits training data into X and y lists, and transforms any necessary data. 

        training_data (pd.DataFrame) : data to prep
        scaler_path (str) : scaler_path to transform X values

        Returns:
        tuple[list, list] : X and y values
        """
        X = training_data.drop(columns = ["HOME_TEAM_WIN"])
        X = X.values
        y = pd.get_dummies(training_data, columns=["HOME_TEAM_WIN"], prefix="Result")[
            "Result_W"
        ].values

        if scaler_path:
            scaler = joblib.load(scaler_path)
            X = scaler.transform(X)
        else:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

        return X, y

    def initialize_nerual_network() -> Sequential:
        """
        Initializes neural network architecture. 

        Returns:
        (Sequential) : initial empty neural network
        """
        INPUT_DIM = 55
        SECOND_LAYER = 28
        THIRD_LAYER = 14
        FOURTH_LAYER = None
        DROPOUT_RATE = .45
        OUTPUT_LAYER = 1
        NN_model = Sequential()
        NN_model.add(
            Dense(
                units=SECOND_LAYER,
                activation="relu",
                input_dim=INPUT_DIM,
            )
        )
        NN_model.add(Dropout(DROPOUT_RATE))
        NN_model.add(Dense(units=THIRD_LAYER, activation="relu"))
        NN_model.add(Dropout(DROPOUT_RATE))
        NN_model.add(
            Dense(
                units=OUTPUT_LAYER,
                activation="sigmoid",
            )
        )
        NN_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        return NN_model
    
    def train_model(X : list, y : list) -> Sequential:
        """
        Model training pipeline.

        X (list) : X training data 
        y (list) : y training data

        Returns:
        (Sequential) : Sequential Keras model 
        """
        NN_model = MLTools.initialize_nerual_network()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.5, random_state=42
        )
        y_train_numerical = y_train.astype(int)
        y_val_numerical = y_val.astype(int)

        # apply principal component analysis 
        pca = PCA(n_components=55)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        X_val = pca.transform(X_val)

        # train model
        history = NN_model.fit(
            X_train,
            y_train_numerical,
            epochs=20,
            batch_size=64,
            validation_data=(X_val, y_val_numerical),
        )

        return NN_model

    def load_model(path : str):
        """
        Loads NN model 

        path (str) : path of model 

        Returns:
        
        """

data = MLTools.load_training_data("./data/")
data = MLTools.preprocess_training_data(data)
X, y = MLTools.prep_data(data)

model = MLTools.train_model(X, y)