from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from src.data.load_data import movies, ratings
import os
import pickle 

class HybridRecommender:
    def __init__(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings
        self.content_sim_matrix = None
        self.tfidf_matrix = None
        self.collab_model = None
        self.trained = False
        
    def train(self):
        # Content-based part (unchanged)
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies['genres'].fillna(''))
        self.content_sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Collaborative filtering part - using scikit-learn instead of surprise
        # Create user-item matrix
        user_item_matrix = self.ratings.pivot_table(
            index='movieId', 
            columns='userId', 
            values='rating'
        ).fillna(0)
        
        # Train KNN model
        self.collab_model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.collab_model.fit(user_item_matrix)
        
        self.trained = True
    
    def recommend(self, user_id, movie_id=None, title=None, top_n=5):
        if not self.trained:
            self.train()
            
        if movie_id is None and title is not None:
            movie_id = self.movies[self.movies['title'] == title]['movieId'].values[0]
        
        # Content-based recommendations (unchanged)
        idx = self.movies[self.movies['movieId'] == movie_id].index[0]
        sim_scores = list(enumerate(self.content_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        content_indices = [i[0] for i in sim_scores[1:top_n+1]]
        content_recs = self.movies.iloc[content_indices][['movieId', 'title']]
        content_recs['type'] = 'Content'
        
        # Collaborative recommendations using scikit-learn
        # Create user-item matrix for the movie
        user_item_matrix = self.ratings.pivot_table(
            index='movieId', 
            columns='userId', 
            values='rating'
        ).fillna(0)
        
        if movie_id in user_item_matrix.index:
            distances, indices = self.collab_model.kneighbors(
                user_item_matrix.loc[movie_id].values.reshape(1, -1), 
                n_neighbors=top_n+1
            )
            similar_movies = user_item_matrix.iloc[indices[0]].index
            collab_recs = self.movies[self.movies['movieId'].isin(similar_movies)][['movieId', 'title']]
            collab_recs = collab_recs[collab_recs['movieId'] != movie_id].head(top_n)
            collab_recs['type'] = 'Collaborative'
        else:
            collab_recs = pd.DataFrame(columns=['movieId', 'title', 'type'])
        
        # Hybrid approach (same as before)
        all_recs = pd.concat([content_recs, collab_recs])
        hybrid_recs = all_recs.groupby(['movieId', 'title']).size().reset_index(name='score')
        hybrid_recs = hybrid_recs.sort_values('score', ascending=False).head(top_n)
        hybrid_recs['type'] = 'Hybrid'
        
        return {
            'content': content_recs,
            'collaborative': collab_recs,
            'hybrid': hybrid_recs
        }
        
    

  

recommender = HybridRecommender(movies, ratings)

MODEL_PATH = '/home/pawan/devlopment/ML_PROJECTS/Recommendation-app/models/hybrid_model.pkl'
def load_recommender_model(model_path="models/production/latest_model.pkl"):
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    full_model_path = os.path.join(project_root, model_path)

    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Model file not found at: {full_model_path}")
    else:
        recommender.train()
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(recommender, f)
    with open(full_model_path, 'rb') as f:
        model = pickle.load(f)  
    return model         
