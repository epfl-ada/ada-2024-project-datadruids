
from datasets import Dataset
from huggingface_hub import HfApi
from datasets import load_dataset
import pandas as pd

def load_data():
    # ## Load HF Datasets
    book_genres = load_dataset("ada-datadruids/book_genres")
    wiki_book_movie_ids_matching = load_dataset("ada-datadruids/wiki_book_movie_ids_matching")
    TMDB_movie_dataset_reduced = load_dataset("ada-datadruids/TMDB_movie_dataset_reduced")
    plot_summaries = load_dataset("ada-datadruids/plot_summaries")
    movies = load_dataset("ada-datadruids/movies")
    movie_metadata = load_dataset("ada-datadruids/movie-metadata")
    merged_movies_books_cleaned = load_dataset("ada-datadruids/merged_movies_books_cleaned")
    CPIAUCNS = load_dataset("ada-datadruids/CPIAUCNS")
    booksummaries_cleaned = load_dataset("ada-datadruids/booksummaries_cleaned")
    booksmovies = load_dataset("ada-datadruids/booksmovies")
    books = load_dataset("ada-datadruids/books")
    bookfilm_summaries_with_similarity_and_sentiment = load_dataset("ada-datadruids/bookfilm_summaries_with_similarity_and_sentiment")


    # ## Refactor Datasets to Dataframes
    book_genres_df = book_genres['train'].to_pandas()
    wiki_book_movie_ids_matching_df = wiki_book_movie_ids_matching['train'].to_pandas()
    TMDB_movie_dataset_reduced_df = TMDB_movie_dataset_reduced['train'].to_pandas()
    plot_summaries_df = plot_summaries['train'].to_pandas()
    movies_df = movies['train'].to_pandas()
    movie_metadata_df = movie_metadata['train'].to_pandas()
    merged_movies_books_cleaned_df = merged_movies_books_cleaned['train'].to_pandas()
    CPIAUCNS_df = CPIAUCNS['train'].to_pandas()
    booksummaries_cleaned_df = booksummaries_cleaned['train'].to_pandas()
    booksmovies_df = booksmovies['train'].to_pandas()
    books_df = books['train'].to_pandas()
    bookfilm_summaries_with_similarity_and_sentiment_df = bookfilm_summaries_with_similarity_and_sentiment['train'].to_pandas()


    # ## Store Dataframes as files in the datafolder
    book_genres_df.to_csv('../../data/book_genres.csv', index=False)
    wiki_book_movie_ids_matching_df.to_csv('../../data/wiki_book_movie_ids_matching.csv', index = False)
    TMDB_movie_dataset_reduced_df.to_csv('../../data/TMDB_movie_dataset_reduced.csv', index=False)
    plot_summaries_df.to_csv('../../data/plot_summaries.txt', index=False)
    movies_df.to_csv('../../data/movies.csv', index=False)
    movie_metadata_df.to_csv('../../data/movie_metadata.tsv','\t', index=False)
    merged_movies_books_cleaned_df.to_csv('../../data/merged_movies_books_cleaned.csv', index=False)
    CPIAUCNS_df.to_csv('../../data/CPIAUCNS.csv', index=False)
    booksummaries_cleaned_df.to_csv('../../data/booksummaries_cleaned.csv', index=False)
    bookfilm_summaries_with_similarity_and_sentiment_df.to_csv('../../data/bookfilm_summaries_with_similarity_and_sentiment.csv', index=False)
    books_df.to_csv('../../data/books.csv', index=False)
    booksmovies_df.to_csv('../../data/booksmovies.csv', index=False)

