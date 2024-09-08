import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_data():
    df = pd.read_csv("SpotifyFeatures.csv")
    print(df)
    # Count number of genres
    genres_number = df['genre'].nunique()
    print(genres_number)

    # Reduce dataframe size (only necessary columns)
    df_genre = df[df['genre'].isin(['Pop', 'Classical'])][['track_name','genre', 'liveness', 'loudness']]

    # labels for the samples
    df_genre['label'] = df_genre['genre'].map({'Pop': 1, 'Classical': 0})

    # Count songs in the two genres
    anzahl_classic = df_genre[df_genre['label'] == 0]['label'].count()
    anzahl_pop = df_genre[df_genre['label'] == 1]['label'].count()
    print("Number classic songs: ", anzahl_classic, " Number pop songs: ", anzahl_pop)

    #Create both numpy arrays for X and y data
    X_array = np.array(df_genre[['liveness', 'loudness']])
    y_array = np.array(df_genre[['label']])

    #Split the data with the same size contribution (stratify)
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, random_state=42, stratify=y_array)

    return X_train, X_test, y_train, y_test, X_array, y_array


