# ada-2024-project-datadruids
ada-2024-project-datadruids created by GitHub Classroom
Readme.md file containing the detailed project proposal (up to 1000 words). Your README.md should contain:

## What book to pick to create the next Lord of the Rings?

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
Revenues from TMDB (2024):
https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies 

All of the datasets can be merged by IMDB (or TMDB) IDs or by their titles. The “CMU Book Summaries” dataset is for the plot similarity analysis, “Book Movie Reviews” helps in matching books to their adaptations and the “Revenues from TMDB” dataset increases the number of revenues that are available.

## Methods
### Language Model
Sentence Transformer embeddings are used to measure the similarity between summaries. Given the model’s input length constraints, each summary is split into chunks with small text overlaps that fit within the model’s limits to minimize semantic information loss. Then, the chunks are embedded and averaged to create a single embedding representing the entire summary. Lastly, cosine similarity is calculated for the book and movie embeddings, with higher scores suggesting closer thematic alignment, which can indicate adaptation fidelity. [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) was selected as model due to its [performance and manageable size](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html).

As above the summaries are divided into chunks that receive a sentiment score (e.g. positive, neutral, negative) which is then averaged and compared. This approach provides a holistic view of sentiment similarity, which is another factor that may impact adaptation success. For similar reasons as with the summary similarity, [siebert/sentiment-roberta-large-english](https://huggingface.co/siebert/sentiment-roberta-large-english) was selected for this task.

### Success metric
The analysis will be performed with regard to two success metrics, box office revenue and reviews. These will firstly be treated separately to reduce potential biases, but can be combined later on to reduce complexity for the reader.

### Search for confounders
Regression models will be used to find which factors correlate the most with movie success. This is done on all movies in the dataset and can be used to identify confounders that are unrelated to whether the movie is based on a book or not. They need to be controlled for further analysis.

### Matching analysis
The success of a movie depends on confounders that are unrelated to whether it is based on a book or not. The benefit of basing a movie on a book can be determined through pairwise matching. A propensity score is calculated based on confounders identified beforehand. Movies with similar propensity scores are then matched with the main difference being whether they are based on a book or not.

### 

## Proposed timeline
1. Clean and Prepare dataset by matching films with books and adding summaries from film and books such as missing box office revenues 
2. Define Methods
3. Assign methods (finished by 13.11)
4. Create relevant Plots (started before 15.11, finished by 6.12)
5. Create Web Site 
6. Storytelling, add interactive features (finished by 18.12)


## Organization within the team:
Luca : Finding suitable language models, work with language models, data visualisations
Danja: Dataset research,  first part of dataset merge, search methods, story telling
Silvan: Second part of dataset merge, readme formatting, confounders identification and matching
Robin: Dataset research, abstract, readme formatting, confounders identification and matching
Franziska: Github organisation, clean jupyter notebooks, create clean python files, conception of website



## Questions for TAs (optional): Add here any questions you have for us related to the proposed project.

# Your project name
This is a template repo for your project to help you organise and document your code better. 
Please use this structure for your project and document the installation, usage and structure as below.

## Quickstart

```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```



### How to use the library
Tell us how the code is arranged, any explanations goes here.



## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```
