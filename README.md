# Finding the next Lord of the Rings: From Book to Cinematic Glory

## Abstract:
Books have been adapted into movies since the dawn of cinema. “Tarzan” alone has been adapted nearly forty times. Many questions arise with this fact. Why have so many directors decided to make a movie about this exact book, rather than choosing one that was not yet adapted into a movie? Is there an actual advantage of doing yet another interpretation of a classic, or are the first adaptations normally the best performing ones? 
This data story will try to educate a director with regards to these common pitfalls and trends when it comes to visualizing the written worlds. This will be done by analyzing reviews, movie / book summaries, and money flows. The obtained results can give the director a glimpse into adaptations from the past, to guide their decisions in book selection, screenwriting and casting. 


## Research Questions:
- How advantageous is it for a movie to be based on a book?
- What features of a book are the most important for a successful movie adaptation?
- How does the similarity of plot and sentiment between a book and its adaptation influence the movie success?
- If there are multiple adaptations of the same book, what main factors set these movies apart? Which ones are most     successful?

## Proposed additional datasets: 
CMU Book Summaries : https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset

Book Movie Reviews : https://www.kaggle.com/datasets/captaindylan/books-movies-reviews?resource=download 

Revenues from TMDB (2024): https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies 

All of the datasets can be merged by IMDB (or TMDB) IDs or by their titles. The “CMU Book Summaries” dataset is for the plot similarity analysis, “Book Movie Reviews” helps in matching books to their adaptations and the “Revenues from TMDB” dataset increases the number of revenues that are available.
The merging process below is executed in the Jupyter Notebook books_movies_cleaning, which produces the final dataset. The final dataset is uploaded on github and used in the results.ipynb file. 

### Merging Process

1. **Initial Data Loading**:  
Since the datasets should not be stored directly in the github, they were uploaded onto huggingface. The file data_loader contains a function that downloads all files and stores them in the folder `data`. The notebook `books_movies_cleaning.ipynb` calls the function before starting the merging process.  

2. **Dataset Merging**:
   - Both `wiki_book_movie_ids_matching.csv` and `movies.csv` originally contained duplicates, which were removed.
   - The `runtime` from `movies.csv` was added to `wiki_book_movie_ids_matching.csv` using `imdbid` as the key.
   - The resulting dataset was merged with `movie.metadata.tsv` on `movie_name` and `movie_year`. Since a single book can correspond to multiple movies, matching by `movie_name` alone was insufficient; therefore, `movie_year` was also used to improve the match.
   - Some duplicates remained after this merge. They were resolved using the unique identifiers `imdbid`, `movie_id`, and `id_goodreads`.  
   - Films based on multiple books were removed, as they did not align with the research questions.
   - 45 films with the same title and release year were manually reviewed to prevent incorrect merges. The movies in `movie.metadata.tsv` were compared with those in `wiki_book_movie_ids_matching.csv` using `runtime` as an additional matching criterion.
   - If `runtime` was insufficient to resolve matches, the TMDB database was referenced to cross-verify entries, ensuring accurate film matches.
   - Revenues were sparse. Therefore, additional revenues were collected from the `TMDB_movie_dataset_reduced.csv` dataset and used to complete the original revenues.
   - Finally, numerical book and movie ratings were added to the dataset

3. **Adding Book Summaries**:  
   - Additional dataset containing book summaries was loaded and merged.


## Methods
### Natural Language Processing
Sentence transformer embeddings are used to measure the similarity between summaries. Given the model's input length constraints, each summary is split into chunks with small text overlaps that fit within the model's limits to minimize semantic information loss. Then, the chunks are embedded and averaged to create a single embedding representing the entire summary. Lastly, cosine similarity is calculated for the book and movie embeddings, with higher scores suggesting closer thematic alignment, which can indicate adaptation fidelity. [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) was selected as model due to its [performance and manageable size](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html).

Similarly, summary sentiments are computed. As for the embeddings, the summaries are divided into chunks that receive a sentiment score which is then averaged and compared. This approach provides a holistic view of sentiment similarity, which is another factor that may impact adaptation success. For similar reasons as with the summary similarity, [siebert/sentiment-roberta-large-english](https://huggingface.co/siebert/sentiment-roberta-large-english) was selected for this task.

### Success metric
The analysis will be performed with regard to two success metrics, box office revenue and reviews. These will firstly be treated separately to reduce potential biases, but can be combined later on to reduce complexity for the reader.

### Search for confounders
Regression models will be used to find which factors correlate the most with movie success. This is done on all movies in the dataset and can be used to identify confounders that are unrelated to whether the movie is based on a book or not. They need to be controlled for further analysis.

### Matching analysis
The success of a movie depends on confounders that are unrelated to whether it is based on a book or not. The benefit of basing a movie on a book can be determined through pairwise matching. A propensity score is calculated based on confounders identified beforehand. Movies with similar propensity scores are then matched with the main difference being whether they are based on a book or not.

### Interpretation and conclusions
Using regression methods, ANOVA and descriptive statistics and keeping in mind the identified cofounders the most important correlations can be found. Analysing these results will hopefully enable meaningful conclusions.

## Proposed timeline
1. Clean and Prepare dataset by matching films with books and adding summaries from film and books such as missing box office revenues 
2. Define Methods
3. Assign methods (finished by 13.11)
4. Create relevant Plots (started before 15.11, finished by 6.12)
5. Create Web Site 
6. Storytelling, add interactive features (finished by 18.12)


## Organization within the team:
- Luca : Finding suitable language models, work with language models, data visualisations, statistical model fitting
- Danja: Dataset research,  first part of dataset merge, search methods, primary data visualization, story telling
- Silvan: Second part of dataset merge, readme formatting, confounders identification and matching
- Robin: Dataset research, abstract, readme formatting, confounders identification and matching
- Franziska: Github organisation, clean jupyter notebooks, create clean python files, conception of website, statistical model fitting
