import pandas as pd
from pymongo import MongoClient

def load_movies_to_mongodb(csv_path="dataset\movies.csv"):
    movies_df = pd.read_csv(csv_path)
    client = MongoClient('localhost', 27017)
    db = client.Movies_DB
    collection = db.Movies_dataset

    # Clear existing data (optional, for fresh runs)
    collection.delete_many({})

    movies_list = []
    for index, row in movies_df.iterrows():
        # Split genres into a list
        genres = row['genres'].split('|')
        movie_doc = {
            "movie_id": int(row['movieId']), # Use int for MovieLens ID
            "title": row['title'],
            "genres": genres,
            # Add more fields if your CSV/dataset has them (e.g., 'overview', 'cast')
        }
        movies_list.append(movie_doc)

    if movies_list:
        collection.insert_many(movies_list)
        print(f"Successfully inserted {len(movies_list)} movies into MongoDB.")
    else:
        print("No movies to insert.")

if __name__ == "__main__":
    # Make sure 'movies.csv' is in the same directory or provide the full path
    load_movies_to_mongodb()