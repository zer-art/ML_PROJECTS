import pandas as pd
import os

def load_data(movie_file_name='movies.dat', rating_file_name='ratings.dat'): # Set defaults for convenience
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    movie_file_path = os.path.join(project_root, 'data', movie_file_name)
    rating_file_path = os.path.join(project_root, 'data', rating_file_name)

    # Load movies data
    movies = pd.read_csv(
        movie_file_path,
        sep='::',
        header=None,
        engine='python', # 'python' engine needed for '::' separator
        names=['movieId', 'title', 'genres'],
        encoding='latin-1' # Crucial for handling non-UTF-8 characters
    )

    # Load ratings data
    ratings = pd.read_csv(
        rating_file_path,
        sep='::',
        header=None,
        engine='python', # 'python' engine needed for '::' separator
        names=['userId', 'movieId', 'rating', 'timestamp'],
        encoding='latin-1' # Crucial for handling non-UTF-8 characters
    )

    return movies, ratings

movies, ratings = load_data()