# Data-Analysis-Masters-Project

## Movie Recommendation System

## About the Project

Our project is a simple Content-Based Movie Recommendation System. We utilize movie content to recommend similar movies based on the selected one.

## About the Dataset

We use two different datasets for recommending movies:

1. tmdb_5000_credits.csv: This dataset contains columns such as movieId, title, cast, and crew information.
2. tmdb_5000_movies.csv: This dataset includes columns like title, movieId, overview, genres, ratings, and more. We discovered this dataset on Kaggle and have attached it in the “dataset” folder. If you need more information about the dataset or cannot find it in the “dataset” folder, you can access it using the following link:

[Movie Recommendation System Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## Working of the Recommendation SystemAbout Working

We explore three different algorithms to determine which one provides the best results:

1. Cosine Similarity
2. Tf-Idf Similarity
3. KNN (K-Nearest Neighbors) along with Tf-Idf Similarity

Among these, we observed that the “KNN along with Tf-Idf Similarity” approach yields the best results. You can verify this claim by either running the Jupyter file “Movie-Recommender-System.ipynb” or using the web application “app.py”.

## Steps to Run the Project

Follow these steps to run the project:

1. Open “Movie-Recommender-System.ipynb” in Jupyter or Jupyter Lab. Ensure that you have all the required libraries installed.
2. Execute all the cells in the notebook.
3. The results of the mentioned algorithms will be stored in files under the “files” folder.
4. Open any text editor of your choice and run the “app.py” file. Before running it, make sure you have all the necessary libraries installed (as used in “app.py”).
5. Open a terminal, navigate to the project path, and run the following command:
   streamlit run app.py
6. This will open a web page in your browser. Select any movie from the dropdown and click the “Recommend” button. It will display recommended movies based on the selected movie.
