# Datasets Explained
Because Data sets should not be loaded onto the github page, they were uploaded on hugging face and are loaded in the beginning of the books_movies_cleaning.ipynb file

### Primary Datasets

1. **CMU Dataset**
   - **movie.metadata.tsv**: *Wikipedia movie ID, Freebase movie ID, Movie name, Movie release date, Movie box office revenue, Movie languages, Movie countries, Movie genres*
   - **plot_summaries.txt**: *Contains movie plot summaries*

2. **Revenues from TMDB (2024)**  
   Source: [TMDB Dataset on Kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
   - **TMDB_movie_dataset_reduced.csv**: *Contains IMDb ID and revenue for 1'127'778 movies. 20'741 movies have non-zero revenue.*

3. **CMU Book Summaries**  
   Source: [CMU Book Summaries on Kaggle](https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset)
   - **booksummaries_cleaned.csv**: *Contains cleaned book summaries.*

4. **Book Movie Reviews**  
   Source: [Books & Movies Reviews on Kaggle](https://www.kaggle.com/datasets/captaindylan/books-movies-reviews?resource=download)
   - **books.csv**: *Includes book metadata (book_id, title, year, average rating, rating count, review count, series, series number, author, description, length, star ratings, cover image, standardized rating, normalized rating).*
   - **movies.csv**: *_path, rating average, vote count, standardized rating, normalized rating, goodreads scale rating*
   - **booksmovies.csv**: *(id, BookTitle, BookAltTitle, BookYear, BookLink, Author, FilmTitle, FilmLink, FilmYear, ISBN)*
   - **merged_movies_books_cleaned.csv**: *(movie_id, movie_cluster, movie_name, release_date, budget, runtime_x, language, country, genres, movie_year, index, id, BookTitle, BookYear, Author, full name, tmdb_id, id_goodreads, imdbid, runtime_y)*
   - **wiki_book_movie_ids_matching.csv**: *(index, id, BookTitle, BookYear, Author, FilmTitle, FilmYear, full name, tmdb_id, id_goodreads, imdbid)*
   - **book_genres.csv**: *(book_id, genre, possible_filter)*
  
5. **CPI Scores**
  Source: [Consumer Price Index U.S.](https://fred.stlouisfed.org/series/CPIAUCNS )
   - **CPIAUCNS.csv**: *DATE,CPIAUCNS*

### Datasets Extracted from Jupyter Notebooks

1. **bookfilm_summaries_with_similarity_and_sentiment.csv**:  
   *(movie_id, book_title, movie_name, book_summary, film_summary, similarity, film_sentiment, film_sentiment_score, book_sentiment, book_sentiment_score)*
2. **final_dataset.csv**:
   *(movie_id,movie_cluster,movie_name,release_date,runtime_x,language,country,genres,movie_year,index,id,BookTitle,BookYear,Author,full name,tmdb_id,id_goodreads,imdbid,runtime_y,revenue,book_id,normalized_rating_x,standardized_rating_x,normalized_rating_y,standardized_rating_y,length,review_count)*

---

### Merging Process

The merging process is detailed in the Jupyter Notebook books_movies_cleaning, which produced the final dataset.

1. **Initial Data Loading**:  
   The files `movie.metadata.tsv`, `movies.csv`, and `wiki_book_movie_ids_matching.csv` were loaded into the Jupyter Notebook. The `movie.metadata.tsv` file contains details about films, while the `wiki_book_movie_ids_matching.csv` file provides film names with corresponding book names.

2. **Duplicate Removal**:  
   - Both `wiki_book_movie_ids_matching.csv` and `movies.csv` originally contained duplicates, which were removed.
   - The `runtime` from `movies.csv` was added to `wiki_book_movie_ids_matching.csv` using `imdbid` as the key.

3. **Dataset Merging**:  
   - The resulting dataset was merged with `movie.metadata.tsv` on `movie_name` and `movie_year`. Since a single book can correspond to multiple movies, matching by `movie_name` alone was insufficient; therefore, `movie_year` was also used to improve the match.
   - Some duplicates remained after this merge. They were resolved using the unique identifiers `imdbid`, `movie_id`, and `id_goodreads`.

4. **Filtering Non-Applicable Films**:  
   - Films based on multiple books were removed, as they did not align with the research questions.

5. **Manual Matching for Accuracy**:  
   - 45 films with the same title and release year were manually reviewed to prevent incorrect merges. The movies in `movie.metadata.tsv` were compared with those in `wiki_book_movie_ids_matching.csv` using `runtime` as an additional matching criterion.
   - If `runtime` was insufficient to resolve matches, the TMDB database was referenced to cross-verify entries, ensuring accurate film matches.

6. **Adding Book Summaries**:  
   - Following the above steps, additional datasets containing book summaries were loaded and merged into the final dataset.
