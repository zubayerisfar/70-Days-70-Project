import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the SVD model and cosine similarity matrix
svd = joblib.load(r'E:\Hybrid Recommendation System\models\svd_model.pkl')  # Load the pre-trained SVD model
cosine_sim = np.load(r'E:\Hybrid Recommendation System\models\cosine_sim.npy')  # Load the cosine similarity matrix

# Assuming the 'movies' DataFrame contains the movie metadata (e.g., movieId and title)
# You can load it from a CSV file if it's not already in memory.
# You can load it from a CSV file if it's not already in memory.
movies = pd.read_csv(r'E:\Hybrid Recommendation System\datasets\movies.csv')

# Hybrid recommendation function (already defined)
def hybrid_recommend(movie_id, svd, cosine_sim, top_n=10, content_weight=0.5, collab_weight=0.5):
    movie_idx = movie_id - 1  # Convert to 0-based index
    
    # Collaborative filtering (SVD) score
    movie_vec = svd.components_[:, movie_idx]  # Extract latent features for the target movie
    collab_score = svd.components_.T.dot(movie_vec)  # Dot product with the transposed matrix
    collab_score = collab_score.flatten()  # Flatten the similarity scores

    # Content-based (Cosine similarity) score
    content_score = cosine_sim[movie_idx]  # Get the similarity scores for the target movie
    
    # Ensure both arrays have the same length
    if len(content_score) > len(collab_score):
        content_score = content_score[:len(collab_score)]
    if len(collab_score) > len(content_score):
        collab_score = collab_score[:len(content_score)]

    hybrid_score = content_weight * content_score + collab_weight * collab_score

    top_movie_indices = hybrid_score.argsort()[-top_n-1:-1][::-1]
    
    recommended_titles = [movies.iloc[i]['title'] for i in top_movie_indices if i != movie_idx]
    return recommended_titles

# Streamlit UI
st.title('Movie Recommendation System')

# Movie selection (dropdown)
movie_choices = movies['title'].tolist()
selected_movie = st.selectbox("Select a Movie", movie_choices)

# Get the movie_id from the selected movie title
movie_id = movies[movies['title'] == selected_movie]['movieId'].values[0]

# Get the top N recommendations for the selected movie
top_n = st.slider('Select the number of recommendations', 1, 20, 10)

# Sliders for content-based and collaborative filtering weight
content_weight = st.slider('Content-based Weight', 0.0, 1.0, 0.5)
collab_weight = st.slider('Collaborative Filtering Weight', 0.0, 1.0, 0.5)

# Display the results
if st.button('Get Recommendations'):
    recommended_movies = hybrid_recommend(movie_id, svd, cosine_sim, top_n=top_n, content_weight=content_weight, collab_weight=collab_weight)
    st.write(f"Recommended Movies similar to '{selected_movie}':")
    for movie in recommended_movies:
        st.write(f"- {movie}")
