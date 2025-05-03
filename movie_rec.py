import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

movies = pd.read_csv(r"D:\projs\TE Projs\NLP mini proj\movie_dataset.csv")

#Extracting list from dictionary of lists by parsing feature
def parse_features(x):
    try:
        return [d['name'] for d in ast.literal_eval(x)]
    except:
        return []
    

def get_director(x):
    try:
        for i in ast.literal_eval(x):
            if i['job'] == 'Director':
                return i['name']
    
    except:
        return ''
    return ''

def group_words(text, group_size=2):
    words = text.split()
    return [' '.join(words[i:i+group_size]) for i in range(0, len(words), group_size)]

# alsp will be pre processing 
movies['cast']=movies['cast'].astype(str)

#main filtering and applying
movies['genres'] = movies['genres'].apply(lambda x: x.split() if isinstance(x, str) else [])
movies['keywords'] = movies['keywords'].apply(lambda x: x.split() if isinstance(x, str) else [])
movies['cast'] = movies['cast'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
movies['crew'] = movies['crew'].apply(get_director)


def create_soup(row):
    return ' '.join(row['keywords']) + ' ' + ' '.join(row['cast']) + ' ' + row['crew'] + ' ' + ' '.join(row['genres'])

movies['soup'] = movies.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix =count.fit_transform(movies['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Reset index for accurate lookup
movies = movies.reset_index()
movies['title_lower'] = movies['title'].str.lower().str.strip()
indices = pd.Series(movies.index, index=movies['title_lower'])

def recommend(title, num_recommendations=5):
    title = title.lower().strip()  # Normalize user input
    if title not in indices:
        return f"Movie '{title}' not found in dataset."
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    rec =  movies[['title', 'vote_average', 'genres']].iloc[movie_indices]
    return rec['title']


print(recommend("The Avengers"))