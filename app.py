# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import sys
import threading # Import threading to ensure one-time execution

# Import functions from your recommender_core.py
from recommender_core import get_movies_dataframe, preprocess_data, get_tfidf_matrix, calculate_cosine_similarity, get_recommendations

app = Flask(__name__)

# Global variables to store our pre-calculated data
movies_df = None
cosine_sim_matrix = None
# A flag to ensure initialization runs only once
_recommender_initialized = False
_recommender_lock = threading.Lock() # To prevent race conditions if multiple requests come in at once

# --- Application Initialization ---
@app.before_request
def initialize_recommender():
    """
    Initializes the recommendation engine before the first request using a lock
    to ensure it runs only once.
    """
    global movies_df, cosine_sim_matrix, _recommender_initialized

    # Use a lock to ensure only one thread initializes the recommender
    with _recommender_lock:
        if not _recommender_initialized:
            print("🎬 Initializing Flask Movie Recommender Engine... 🍿")

            # Step 1: Get DataFrame
            movies_df = get_movies_dataframe()
            if movies_df.empty:
                print("ERROR: No movie data available. Please ensure your MongoDB has data and is running.")
                sys.exit(1) # Exit if no data, as the app won't function

            # Step 2: Preprocess data
            movies_df = preprocess_data(movies_df)

            # Step 3: Get TF-IDF matrix
            tfidf_matrix, _ = get_tfidf_matrix(movies_df)

            # Step 4: Calculate Cosine Similarity
            cosine_sim_matrix = calculate_cosine_similarity(tfidf_matrix)

            _recommender_initialized = True
            print("✅ Movie Recommender Engine initialized successfully!")

# --- Routes ---
@app.route('/')
def index():
    """
    Renders the main page of the application.
    """
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Handles the recommendation request from the web form.
    """
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        
        if not movie_title:
            return jsonify({'error': 'Please enter a movie title.'}), 400

        # Ensure that movies_df and cosine_sim_matrix are loaded
        # This check is largely redundant now due to before_request, but good for safety
        if movies_df is None or cosine_sim_matrix is None:
            return jsonify({'error': 'Recommender engine not initialized. Please restart the server.'}), 500

        recommendations = get_recommendations(movie_title, cosine_sim_matrix, movies_df)

        if recommendations:
            return jsonify({'recommendations': recommendations})
        else:
            return jsonify({'message': f"Could not find recommendations for '{movie_title}'. Please try another title or check your spelling."})

if __name__ == '__main__':
    # Make sure you have Flask installed: pip install Flask
    # Ensure your recommender_core.py is in the same directory or properly set in PYTHONPATH
    app.run(debug=True) # debug=True allows auto-reloading and better error messages