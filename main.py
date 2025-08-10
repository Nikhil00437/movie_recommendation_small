import pandas as pd
from recommender_core import get_movies_dataframe, preprocess_data, get_tfidf_matrix, calculate_cosine_similarity, get_recommendations
import sys

def main():
    """
    Main function to run the movie recommendation system as a CLI application.
    Initializes the recommendation engine once and then allows multiple queries.
    """
    print("🎬 Welcome to the Movie Recommender! 🍿")
    print("Initializing recommendation engine... This might take a moment.")

    # Step 1: Get DataFrame
    movies_df = get_movies_dataframe()
    if movies_df.empty:
        print("Exiting. No movie data available. Please ensure your MongoDB has data.")
        sys.exit(1) # Exit if no data

    # Step 2: Preprocess data
    movies_df = preprocess_data(movies_df)

    # Step 3: Get TF-IDF matrix
    tfidf_matrix, _ = get_tfidf_matrix(movies_df)

    # Step 4: Calculate Cosine Similarity
    cosine_sim = calculate_cosine_similarity(tfidf_matrix)

    print("\nEngine ready! You can now ask for movie recommendations.")
    print("Type 'exit' or 'quit' to stop the program.")

    while True:
        user_input = input("\nEnter a movie title (or 'exit'): ").strip()

        if user_input.lower() in ['exit', 'quit']:
            print("Thanks for using the Movie Recommender. Goodbye! 👋")
            break

        if not user_input:
            print("Please enter a movie title.")
            continue

        # Step 5: Get Recommendations
        recommendations = get_recommendations(user_input, cosine_sim, movies_df)

        if recommendations:
            print("\nHere are your recommendations:")
            for i, movie in enumerate(recommendations):
                print(f"{i+1}. {movie}")
        else:
            print("Could not find recommendations for that movie. Please try another title or check your spelling.")

if __name__ == "__main__":
    main()