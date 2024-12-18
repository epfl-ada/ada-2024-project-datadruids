import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import plotly.express as px
import statsmodels.api as sm


def multi_one_hot_encode_columns(data_df, column_names_mohe, column_names_ohe = None, threshold=150):
    if column_names_ohe: 
        data_df = pd.get_dummies(data_df, columns=column_names_ohe, drop_first=True)

    for column_name in column_names_mohe:
        data_df[column_name] = data_df[column_name].apply(
            lambda x: x.split(', ') if x is not None and len(x) > 0 else [])

        data_df[f'{column_name}_count'] = data_df[column_name].apply(lambda x: len(x))

        # Count the occurrences of each element
        element_counts = pd.Series([item for sublist in data_df[column_name] for item in sublist]).value_counts()

        # Filter elements that appear more than threshold times
        frequent_elements = element_counts[element_counts > threshold].index

        # Filter the column to only include frequent elements
        data_df[column_name] = data_df[column_name].apply(
            lambda x: [item for item in x if item in frequent_elements])

        mlb = MultiLabelBinarizer()
        a = mlb.fit_transform(data_df[column_name].to_numpy())
        df_ohe = pd.DataFrame(a, data_df.index, columns=mlb.classes_)
        df_ohe.columns = [f'{column_name}_' + col for col in df_ohe.columns]
        data_df = pd.concat([data_df, df_ohe], axis=1)
        data_df.drop(columns=[column_name], inplace=True)

    return data_df


def adjust_for_inflation(data_df, target_columns, is_plotting_enabled=False):
    inflation = pd.read_csv('./../data/CPIAUCNS.csv', parse_dates=['DATE'])

    inflation['Year'] = inflation['DATE'].dt.year

    inflation_yearly = inflation.groupby('Year')['CPIAUCNS'].mean().reset_index()
    inflation_yearly.columns = ['Year', 'CPI']

    data_df['release_date'] = pd.to_datetime(data_df['release_date'], format='mixed')
    base_year = data_df.release_date.dt.year.max()

    data_df['movie_year'] = data_df.release_date.dt.year
    data_df = data_df[data_df['movie_year'] > 1915]
    base_cpi = inflation_yearly[inflation_yearly['Year'] == base_year]['CPI'].values[0]

    data_df = data_df.merge(inflation_yearly, left_on='movie_year', right_on='Year', how='left')

    for target_column in target_columns:
        adjusted_name = 'adjusted_' + target_column

        data_df[adjusted_name] = data_df[target_column] * (base_cpi / data_df['CPI'])

        if is_plotting_enabled:
            fig = px.scatter(
                title=f"Original vs. Inflation-Adjusted {target_column} (base year = 2013) of Movies"
            )

            fig.add_scatter(
                x=data_df['movie_year'],
                y=data_df[target_column],
                mode='markers',
                name=f'Original {target_column}',
                marker=dict(color='blue'),
                hovertext=data_df['title']
            )

            fig.add_scatter(
                x=data_df['movie_year'],
                y=data_df[adjusted_name],
                mode='markers',
                name=adjusted_name,
                marker=dict(color='red'),
                hovertext=data_df['title']
            )

            # Calculate the mean revenue and adjusted revenue per year
            average_revenue = data_df.groupby('movie_year')[target_column].mean()
            average_adjusted_revenue = data_df.groupby('movie_year')[adjusted_name].mean()

            # Add trend lines for the average revenues per year
            fig.add_scatter(
                x=average_revenue.index,
                y=average_revenue,
                mode='lines',
                name=f'Average Original {target_column}',
                line=dict(color='black', dash='dash'),
            )

            fig.add_scatter(
                x=average_adjusted_revenue.index,
                y=average_adjusted_revenue,
                mode='lines',
                name=f'Average {adjusted_name}',
                line=dict(color='black', dash='dot'),
            )

            fig.update_layout(
                xaxis=dict(title='Release Year'),
                yaxis=dict(type='log', title=f'{target_column} (USD), log scale'),
                showlegend=True,
                width=800,
                height=500
            )

            fig.show()

    return data_df.drop(columns=['Year', 'CPI'])


def create_train_test_split(data_df, target_column, should_split_based_on_book=False, test_size=0.2, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    data_df_dict = {}
    if should_split_based_on_book:
        based_on_book_df = data_df[data_df['based_on_book'] == True]
        not_based_on_book_df = data_df[data_df['based_on_book'] == False]
        data_df_dict['based_on_book'] = based_on_book_df.drop(columns=['based_on_book'])
        data_df_dict['not_based_on_book'] = not_based_on_book_df.drop(columns=['based_on_book'])

    else:
        data_df_dict["all"] = data_df

    split_dict = {}
    for key, data_df in data_df_dict.items():
        X = data_df.drop(columns=[target_column])
        y = data_df[target_column]

        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=test_size, random_state=seed)

        scaler = StandardScaler()
        scaler.fit_transform(X_train_raw)

        X_train_standardized = scaler.transform(X_train_raw)
        X_test_standardized = scaler.transform(X_test_raw)

        X_train_standardized = pd.DataFrame(X_train_standardized, columns=list(X.columns))
        X_test_standardized = pd.DataFrame(X_test_standardized, columns=list(X.columns))

        X_train = sm.add_constant(X_train_standardized, has_constant = 'add')
        X_test = sm.add_constant(X_test_standardized, has_constant = 'add')
        y_train = y_train_raw.values
        y_test = y_test_raw.values

        split_dict[key] = (X_train, X_test, y_train, y_test)

    return split_dict

def load_books_movies(path_final_dataset = '../data/final_dataset.csv' ):
    # the dataframe loaded is the final dataset, see notebook books_movies_cleaning.ipynb to see how books and movies are matched
    df_books_movies = pd.read_csv(path_final_dataset)
    df_books_movies.drop(columns=['Unnamed: 0'], inplace=True)
    df_books_movies.dropna(subset = ['revenue', 'budget'], inplace=True)
    df_books_movies = df_books_movies[df_books_movies['revenue']!=0]
    df_books_movies = df_books_movies[df_books_movies['budget']!=0]
    df_books_movies.rename(columns={'movie_name': 'title'}, inplace=True)
    return df_books_movies

def merge_and_clean_sim(books_movies_adj):
    similarities = pd.read_csv('../data/bookfilm_summaries_with_similarity_and_sentiment.csv')
    merged_df = books_movies_adj.merge(similarities, how = 'left', on= 'movie_id')

    # Group by 'book_id' and 'movie_id', then calculate the mean for 'similarity' and 'book_sentiment_score' to get rid of doubled columns for similarity scores
    averaged_df = merged_df.groupby(['book_id', 'movie_id'], as_index=False).agg({
        'similarity': 'mean',
        'book_sentiment_score': 'mean',
    })
    merged_df = merged_df.drop(columns=['similarity', 'book_sentiment_score'])
    merged_df.drop_duplicates(subset=['movie_id', 'book_id'], inplace=True)

    merged_df = merged_df.merge(averaged_df, how='left', on=['book_id', 'movie_id'])

    #We have 259 Nan values for the sentiment, what to do with those? drop for now
    merged_df.dropna(subset= ['similarity'], inplace = True)
    
    return merged_df

def books_features_preprocessing(book_sim_genres):
    books_df = pd.read_csv('../data/books.csv')

    books_df_final = book_sim_genres.merge(books_df, how = 'left', on='book_id')
    books_df_final['part_of_series'] = books_df_final['series'].notna().astype(int)
    rating_columns = ['one_star', 'two_stars', 'three_stars', 'four_stars', 'five_stars']
    for col in rating_columns:
        percentage_col = col + '_percentage'
        books_df_final[percentage_col] = books_df_final[col] / books_df_final['rating_count']
    books_df_final.dropna(subset = ['length'], inplace=True)
    return books_df_final