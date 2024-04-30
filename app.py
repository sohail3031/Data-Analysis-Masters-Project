import streamlit as st
import pandas as pd

import pickle
import requests

from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# load data from files
movies_dict = pickle.load(open("files/movies.pkl", "rb"))
movies = pd.DataFrame(movies_dict)
new_movies_df_dict = pickle.load(open("files/new_movies_df.pkl", "rb"))
new_movies_df = pd.DataFrame(new_movies_df_dict)
# similarity = pickle.load(open("similarity.pkl", "rb"))
similarity = pickle.load(open("files/spm.pkl", "rb"))
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(new_movies_df['tags'])
model = NearestNeighbors().fit(tfidf_matrix)


# method to fetch the poster from web
def fetch_poster(movie_id):
    try:
        response = \
            requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&"
                         f"language=en-US").json()['poster_path']

        return f"https://image.tmdb.org/t/p/w500{response}"
    except Exception as e:
        response = requests.get(f"https://api.themoviedb.org/3/movie/{19995}?api_key=8265bd1679663a7ea12ac168da84d2e8&"
                                f"language=en-US").json()['poster_path']

        return f"https://image.tmdb.org/t/p/w500{response}"


# method to recommend movies based on the selected movie using tf-idf and knn
def recommender_knn(movie_name):
    idx = process.extractOne(movie_name, new_movies_df["title"])[2]
    movie_vector = tfidf_matrix[idx]
    distances, indices = model.kneighbors(movie_vector, n_neighbors=20)
    recommended_list, recommended_movies_posters_list, popularity, runtime, vote_average, vote_count = [], [], [], [], [], []

    for i in indices.flatten():
        if i != idx:
            recommended_list.append(new_movies_df.iloc[i]["title"])
            recommended_movies_posters_list.append(fetch_poster(new_movies_df.iloc[i]["movie_id"]))
            popularity.append(movies.iloc[i]["popularity"])
            runtime.append(movies.iloc[i]["runtime"])
            vote_average.append(movies.iloc[i]["vote_average"])
            vote_count.append(movies.iloc[i]["vote_count"])

    return recommended_list[1: 6], recommended_movies_posters_list[1: 6], popularity[1: 6], runtime[1: 6], vote_average[1: 6], vote_count[1: 6]


# code to display the recommended movies on a web page
st.title("Movie Recommender System")

selected_movie_name = st.selectbox("Search Movies Here", movies["title"].values)

if st.button("Recommend"):
    movies_name, movies_posters, popularity, runtime, vote_average, vote_count = recommender_knn(selected_movie_name)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.subheader(movies_name[0])
        st.image(movies_posters[0], use_column_width=True)
        st.text(f"Popularity: {popularity[0]}")
        st.text(f"Run Time: {runtime[0]}")
        st.text(f"Vote Average: {vote_average[0]}")
        st.text(f"Vote Count: {vote_count[0]}")
    with col2:
        st.subheader(movies_name[1])
        st.image(movies_posters[1], use_column_width=True)
        st.text(f"Popularity: {popularity[1]}")
        st.text(f"Run Time: {runtime[1]}")
        st.text(f"Vote Average: {vote_average[1]}")
        st.text(f"Vote Count: {vote_count[1]}")
    with col3:
        st.subheader(movies_name[2])
        st.image(movies_posters[2], use_column_width=True)
        st.text(f"Popularity: {popularity[2]}")
        st.text(f"Run Time: {runtime[2]}")
        st.text(f"Vote Average: {vote_average[2]}")
        st.text(f"Vote Count: {vote_count[2]}")
    with col4:
        st.subheader(movies_name[3])
        st.image(movies_posters[3], use_column_width=True)
        st.text(f"Popularity: {popularity[3]}")
        st.text(f"Run Time: {runtime[3]}")
        st.text(f"Vote Average: {vote_average[3]}")
        st.text(f"Vote Count: {vote_count[3]}")
    with col5:
        st.subheader(movies_name[4])
        st.image(movies_posters[4], use_column_width=True)
        st.text(f"Popularity: {popularity[4]}")
        st.text(f"Run Time: {runtime[4]}")
        st.text(f"Vote Average: {vote_average[4]}")
        st.text(f"Vote Count: {vote_count[4]}")
