# Movie-recommendation-system

A simple and effective content-based movie recommendation system built with Python. It recommends similar movies based on metadata such as genres, cast, crew, keywords, and more using cosine similarity.

Features

Recommends top 5 similar movies based on user input
Utilizes cosine similarity on combined textual features
Fast and lightweight (runs locally without any external API)
Based on the popular TMDB dataset
Easily extendable to include GUI or web interface
Dataset

This project uses the TMDB 5000 Movie Dataset from Kaggle, which includes:

Titles
Overview
Genres
Keywords
Cast and Crew
How It Works

Combine relevant textual metadata for each movie.
Convert the combined text into numerical vectors using CountVectorizer or TfidfVectorizer.
Calculate cosine similarity between all movie vectors.
Recommend top N similar movies to the selected title.
Technologies Used

Python 3.x
Pandas
NumPy
Scikit-learn
Prerequisites

Make sure Python 3.x and pip are installed.
