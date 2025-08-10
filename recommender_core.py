import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re # For cleaning movie titles if needed

# In recommender_core.py, modify get_movies_dataframe()
def get_movies_dataframe():
    client = MongoClient('localhost', 27017)
    db = client.Movies_DB
    collection = db.Movies

    print("Fetching movies from MongoDB...")

    # --- ADD THIS LINE TO LIMIT THE NUMBER OF MOVIES ---
    # For example, to use only the first 10,000 movies
    movies_data = list(collection.find({}).limit(10000)) 
    # For 10,000 movies, the matrix would be 10000*10000*8 bytes = 0.8 GB (manageable)

    if not movies_data:
        print("No movies found in the collection. Please ensure Phase 1 data ingestion was successful.")
        return pd.DataFrame()

    movies_df = pd.DataFrame(movies_data)
    # ... rest of your code ...
    # Drop the MongoDB _id column as it's not needed for recommendations
    if '_id' in movies_df.columns:
        movies_df = movies_df.drop(columns=['_id'])

    print(f"Loaded {len(movies_df)} movies from MongoDB.")
    return movies_df

# --- 2. Data Preprocessing for TF-IDF ---
def preprocess_data(df):
    """
    Preprocesses the DataFrame for TF-IDF vectorization.
    For MovieLens, it joins genres into a single string.
    For TMDB, it would combine 'overview', 'genres', 'cast', 'crew'.
    """
    # Assuming MovieLens structure for now, where 'genres' is a list
    # If your TMDB data has 'overview', 'keywords', 'cast', 'crew', you'd combine those here.
    df['combined_features'] = df['genres'].apply(lambda x: ' '.join(x).replace(' ', '') if isinstance(x, list) else str(x).replace(' ', ''))
    
    # You might want to clean the title to match user input more easily
    df['clean_title'] = df['title'].apply(lambda x: re.sub(r'\s*\(\d{4}\)$', '', x).strip().lower() if isinstance(x, str) else x)
    
    print("Data preprocessing complete.")
    return df

# --- 3. Feature Extraction (TF-IDF) ---
def get_tfidf_matrix(df):
    """
    Applies TF-IDF vectorization to the combined features of the movies.
    """
    tfidf = TfidfVectorizer(stop_words='english') # 'stop_words' is useful if you have plot summaries
    print("Generating TF-IDF matrix...")
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    return tfidf_matrix, tfidf # Return tfidf object for potential later use if needed

# --- 4. Similarity Calculation (Cosine Similarity) ---
def calculate_cosine_similarity(tfidf_matrix):
    """
    Calculates the cosine similarity between all movie feature vectors.
    """
    print("Calculating cosine similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("Cosine similarity matrix calculated.")
    return cosine_sim

# --- 5. Recommendation Function ---
def get_recommendations(movie_title, cosine_sim_matrix, df, top_n=10):
    """
    Generates movie recommendations based on a given movie title.

    Args:
        movie_title (str): The title of the movie to get recommendations for.
        cosine_sim_matrix (np.array): The pre-calculated cosine similarity matrix.
        df (pd.DataFrame): The DataFrame containing movie data.
        top_n (int): The number of recommendations to return.

    Returns:
        list: A list of recommended movie titles.
    """
    clean_input_title = movie_title.lower()
    
    # Create a mapping from clean title to original DataFrame index
    # We use .reset_index() here to get the original row index from the DataFrame
    indices = pd.Series(df.index, index=df['clean_title']).drop_duplicates()

    if clean_input_title not in indices:
        # Try a partial match if exact match fails
        matching_titles = df[df['clean_title'].str.contains(clean_input_title, na=False, regex=False)]
        if not matching_titles.empty:
            print(f"Did not find exact match for '{movie_title}'. Suggesting based on '{matching_titles['title'].iloc[0]}'.")
            # Use the first partial match for recommendation
            movie_idx = indices[matching_titles['clean_title'].iloc[0]]
            original_title_used = matching_titles['title'].iloc[0]
        else:
            print(f"Movie '{movie_title}' not found in the database. Please check the spelling.")
            return []
    else:
        movie_idx = indices[clean_input_title]
        original_title_used = df.loc[movie_idx, 'title']


    # Get the pairwise similarity scores for all movies with that movie
    sim_scores = list(enumerate(cosine_sim_matrix[movie_idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top_n most similar movies (excluding itself)
    sim_scores = sim_scores[1:top_n+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the original titles of the recommended movies
    recommended_titles = df['title'].iloc[movie_indices].tolist()
    print(f"\nRecommendations for '{original_title_used}':")
    return recommended_titles

# --- Main execution flow for testing (optional, can be moved to main.py) ---
if __name__ == "__main__":
    # 1. Get DataFrame
    movies_df = get_movies_dataframe()

    if not movies_df.empty:
        # 2. Preprocess data
        movies_df = preprocess_data(movies_df)

        # 3. Get TF-IDF matrix
        tfidf_matrix, _ = get_tfidf_matrix(movies_df)

        # 4. Calculate Cosine Similarity
        cosine_sim = calculate_cosine_similarity(tfidf_matrix)

        # 5. Get Recommendations (example usage)
        print("\n--- Testing Recommendation Function ---")
        
        # Test with a known movie from MovieLens Small
        example_movie_title_1 = "Toy Story (1995)" 
        recommendations_1 = get_recommendations(example_movie_title_1, cosine_sim, movies_df)
        for i, movie in enumerate(recommendations_1):
            print(f"{i+1}. {movie}")

        print("\n-------------------------------------")
        # Test with another movie
        example_movie_title_2 = "Blade Runner (1982)"
        recommendations_2 = get_recommendations(example_movie_title_2, cosine_sim, movies_df)
        for i, movie in enumerate(recommendations_2):
            print(f"{i+1}. {movie}")

        print("\n-------------------------------------")
        # Test with a movie that might not exist or has a slight typo
        example_movie_title_3 = "NonExistent Movie 123"
        recommendations_3 = get_recommendations(example_movie_title_3, cosine_sim, movies_df)
        if not recommendations_3:
            print("No recommendations found for the non-existent movie as expected.")

        print("\n-------------------------------------")
        # Test with a partial match
        example_movie_title_4 = "lion king" # Should match "Lion King, The (1994)"
        recommendations_4 = get_recommendations(example_movie_title_4, cosine_sim, movies_df)
        for i, movie in enumerate(recommendations_4):
            print(f"{i+1}. {movie}")
    else:
        print("Cannot run recommendation core without movie data. Please load data first.")