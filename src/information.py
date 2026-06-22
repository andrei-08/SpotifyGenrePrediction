import numpy as np
import pandas as pd

# load dataset
def load_dataset(path):
    df = pd.read_csv(path)
    return df

loaded_df = load_dataset("../dataset.csv")

# Find unique values in track_genre column
unique_genres = loaded_df["track_genre"].unique()
print("Unique genres in the dataset:")
for genre in unique_genres:
    print(genre)

