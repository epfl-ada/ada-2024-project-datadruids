# Datasets explained

### CMU_Dataset: 
movie.metadata.tsv: Wikipedia movie ID, Freebase movie ID, Movie name, Movie release date, Movie box office revenue, Movie languages, Movie countries, Movie genres
plot_summaries.txt: contains movie plot summaries 

### Revenues from TMDB (2024):
https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies 
TMDB_movie_dataset_reduced.csv: contains the imdbid and revenue for 1127778 movies, 20741 movies have non zero revenue

### CMU Book Summaries : https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset: 
booksummaries_cleaned.csv: contains booksummaries

Book Movie Reviews : https://www.kaggle.com/datasets/captaindylan/books-movies-reviews?resource=download 

books.csv: (book_id,title,year,avg_rating,rating_count,review_count,series,series_num,author,description,length,five_stars,four_stars,three_stars,two_stars,one_star,cover_image,standardized_rating,normalized_rating)
movies.csv: (_path,rating_average,vote_count,standardized_rating,normalized_rating,goodreads_scale_rating)
booksmovies.csv: (id,BookTitle,BookAltTitle,BookYear,BookLink,Author,FilmTitle,FilmLink,FilmYear,ISBN)
merged_movies_books_cleaned.csv: (movie_id,movie_cluster,movie_name,release_date,budget,runtime_x,language,country,genres,movie_year,index,id,BookTitle,BookYear,Author,full name,tmdb_id,id_goodreads,imdbid,runtime_y)
wiki_book_movie_ids_matching.csv: index,id,BookTitle,BookYear,Author,FilmTitle,FilmYear,full name,tmdb_id,id_goodreads,imdbid

### Datasets extracted from our jupyter notebooks: 
bookfilm_summaries_with_similarity_and_sentiment.csv: (movie_id, book_title, movie_name, book_summary, film_summary, similarity, film_sentiment, film_sentiment_score, book_sentiment, book_sentiment_score)

The merging process can be seen in … which resulted in the final data set … 
First the movie.metadata.tsv, movies.csv and the wiki_book_movie_ids_matching.csv files were loaded into the jupyter notebook. the movie metadata contains the films and the wiki book contains the film names with corresponding book names. 
The wiki_book_movie_ids_matching.csv and movies.csv originally contained duplicates which had to be removed. The runtime from the movies.csv were added to the wiki_book_movie_ids_matching.csv using the imdbid. 
The resulting dataset was merged with the movie.metadata.tsv on movie_name and movie_year. Because there can be more than one movie with the same name for the same book, the movie name was not sufficient to merge therefore the year was used additionally. 
There were still some duplicates present, which were removed using imdbid movie id id goodreads which are the unique identifiers for each dataset.. 
the films which used one or more books used for a film were dropped because it does not align with the research questions. 
45 films with the same name have been released in the same year. To make sure that the 
films in movie metadata were not merged with the wrong film in the wiki book dataset, they were compared by hand using the runtimes in movie metadata and runtimes added from movies to wiki. If that did not suffice, the tmdb database was used to compare other entries. Therefore movie matches could be corrected if necessary. 
Following this the dataset … were loaded which contains the book summaries. They were 




