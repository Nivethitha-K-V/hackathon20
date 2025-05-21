import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df = df[['title', 'genres']].dropna()
    return df

movies = load_data()

# Preprocess
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommend movies
def recommend(title):
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_name = st.text_input("Enter a movie title:")

if movie_name:
    results = recommend(movie_name)
    if not results:
        st.error("Movie not found. Try another title.")
    else:
        st.subheader("You might also like:")
        for movie in results:
            st.write(f"👉 {movie}")
