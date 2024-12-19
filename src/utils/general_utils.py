import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import KFold


def load_final_dataset(path_final_dataset = './data/final_dataset.csv' ):
    # the dataframe loaded is the final dataset, see notebook books_movies_cleaning.ipynb to see how books and movies are matched
    df_books_movies = pd.read_csv(path_final_dataset)
    df_books_movies.drop(columns=['Unnamed: 0'], inplace=True)
    df_books_movies.dropna(subset = ['revenue', 'budget'], inplace=True)
    df_books_movies = df_books_movies[df_books_movies['revenue']!=0]
    df_books_movies = df_books_movies[df_books_movies['budget']!=0]
    return df_books_movies


# def adjust_for_inflation_final_dataset(data, df_revenue) :
#     inflation = pd.read_csv('./../data/CPIAUCNS.csv', parse_dates=['DATE'])

#     inflation['Year'] = inflation['DATE'].dt.year

#     inflation_yearly = inflation.groupby('Year')['CPIAUCNS'].mean().reset_index()
#     inflation_yearly.columns = ['Year', 'CPI']

#     base_year = data.movie_year.max()
#     base_cpi = inflation_yearly[inflation_yearly['Year'] == base_year]['CPI'].values[0]

#     df_revenue = df_revenue.merge(inflation_yearly, left_on='movie_year', right_on='Year', how='left')

#     df_revenue['AdjustedRevenue'] = df_revenue['revenue'] * (base_cpi / df_revenue['CPI'])
    
#     return df_revenue

def adjust_for_inflation(data_df, target_columns, cpiaucns_path='./data/CPIAUCNS.csv', title='title',is_plotting_enabled=False):
    inflation = pd.read_csv(cpiaucns_path, parse_dates=['DATE'])

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
                hovertext=data_df[title]
            )

            fig.add_scatter(
                x=data_df['movie_year'],
                y=data_df[adjusted_name],
                mode='markers',
                name=adjusted_name,
                marker=dict(color='red'),
                hovertext=data_df[title]
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

def px_scatter_plot(data, name_x_values, name_y_values, hover_name, title, labels, template='plotly_white', width=690, height=500, xaxis_type = 'linear',yaxis_type='linear', update_traces=False, hover_data =None) :

    fig = px.scatter(
        data,
        x=name_x_values,
        y=name_y_values,
        hover_name=hover_name,
        title=title,
        labels=labels,
        template=template,
        hover_data=hover_data
    )

    fig.update_layout(
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
        width=width, 
        height=height
    )
    if(update_traces) :
        fig.update_traces(
        hovertemplate='<b>Book Title:</b> %{customdata[0]}<br>' +
                    '<b>Book Release Year:</b> %{customdata[1]}<br>' +
                    '<b>Movie Release Year:</b> %{x}<br>' +
                    '<b>Years to Movie Release:</b> %{y}<br>'
        )

    fig.show()




def assign_quadrant(row, revenue_median, rating_median, revenue_column):
    if row[revenue_column] < revenue_median and row['normalized_rating_x'] < rating_median:
        return 'Low Rating & Low Revenue'
    elif row[revenue_column] < revenue_median and row['normalized_rating_x'] >= rating_median:
        return 'High Rating & Low Revenue'
    elif row[revenue_column] >= revenue_median and row['normalized_rating_x'] < rating_median:
        return 'Low Rating & High Revenue'
    else:
        return 'High Rating & High Revenue'
    

def quadrant_revenue(df_revenue, revenue_column, title, width=900, height=500):

    revenue_median = df_revenue[revenue_column].median()
    rating_median = df_revenue['normalized_rating_x'].median()

    df_revenue['Success Cathegory'] = df_revenue.apply(lambda row: assign_quadrant(row, revenue_median, rating_median, revenue_column), axis=1)

    fig = px.scatter(
        df_revenue,
        x=revenue_column,
        y='normalized_rating_x',
        hover_name='movie_name',
        color='Success Cathegory',
        title= title,
        labels={revenue_column: 'Movie Revenue ($), logscale', 'normalized_rating_x': 'Normalized Movie Rating'},
        template='plotly_white'
    )

    fig.add_shape(
        type='line',
        x0=revenue_median, x1=revenue_median,
        y0=df_revenue['normalized_rating_x'].min(), y1=df_revenue['normalized_rating_x'].max(),
        line=dict(color='Gray', width=1, dash="dash")
    )

    fig.add_shape(
        type='line',
        x0=df_revenue[revenue_column].min(), x1=df_revenue[revenue_column].max(),
        y0=rating_median, y1=rating_median,
        line=dict(color='Gray', width=1, dash="dash")
    )

    fig.update_layout(
        xaxis_type="log",
        width=width,
        height=height
    )

    fig.show()


def plot_regression(results, title) :
    to_include = results.params[results.pvalues < 0.10][1:].sort_values() # get only those with significant pvalues
    fig, ax = plt.subplots(figsize=(5,6), dpi=100)
    ax.scatter(to_include, range(len(to_include)), color="#1a9988", zorder=2)
    ax.set_yticks(range(len(to_include)), to_include.index) # label the y axis with the ind. variable names
    ax.set_xlabel("Proportional Effect")
    ax.set_title(title)

    # add the confidence interval error bars
    for idx, ci in enumerate(results.conf_int().loc[to_include.index].iterrows()):
        ax.hlines(idx, ci[1][0], ci[1][1], color="#eb5600", zorder=1, linewidth=3)

    plt.axline((0,0), (0,1), color="#eb5600", linestyle="--")
    plt.show()

def find_best_params_random_forest(X_train, y_train, param_grid) :
    model = RandomForestRegressor(random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train.ravel())
    best_params = grid_search.best_params_
    print('Best parameters:', best_params)
    return best_params


def standardize_dataset_matching(regression_dataset_processed_df) :
    
    columns = ['vote_count', 'movie_year', 'adjusted_budget', 'adjusted_revenue', 'runtime', 'popularity', 'vote_average', 'genres_Adventure', 'genres_count']
    scaler = StandardScaler()
    scaler.fit_transform(regression_dataset_processed_df[columns])

    regression_dataset_normalized= pd.DataFrame(scaler.transform(regression_dataset_processed_df[columns]), columns=columns)
    regression_dataset_normalized["id"] = regression_dataset_processed_df["id"]
    regression_dataset_normalized["notnorm_revenue"] = regression_dataset_processed_df["adjusted_revenue"]
    regression_dataset_normalized["notnorm_budget"] = regression_dataset_processed_df["adjusted_budget"]
    regression_dataset_normalized["based_on_book"] = regression_dataset_processed_df["based_on_book"]
    regression_dataset_normalized["title"] = regression_dataset_processed_df["title"]

    print(regression_dataset_normalized['based_on_book'].sum())

    X_not_book = regression_dataset_normalized[regression_dataset_normalized['based_on_book'] == False].copy().reset_index(drop=True)
    X_book = regression_dataset_normalized[regression_dataset_normalized['based_on_book'] == True].copy().reset_index(drop=True)

    return X_not_book, X_book


def calculate_propensity_score(data, coeffs) :
    return (coeffs[0] * data["vote_count"] + coeffs[1] * data["movie_year"] + coeffs[2] * data["adjusted_budget"] + coeffs[3] * data["runtime"] + coeffs[4] * data["popularity"] + coeffs[5] * data["vote_average"] + coeffs[6] * data["genres_Adventure"] + coeffs[7] * data["genres_count"])


def matching(X_not_book, X_book ) :
    X_not_book_drop = X_not_book.copy()
    pairs = []
    for index in range(len(X_book)):
        df_sort = X_not_book_drop.iloc[(X_not_book_drop['Propensity']-X_book["Propensity"][index]).abs().argsort()[:2]]
        df_sort = abs(X_book['Propensity'][index] - df_sort["Propensity"])
        df_sort.sort_values(inplace=True)
        if df_sort[df_sort.index[0]] / abs(X_book["Propensity"][index]) <= 0.001:
                pairs.append([index, df_sort.index[0]])
                X_not_book_drop.drop(df_sort.index[0], inplace=True)
            

    return pairs

def create_histogram_matching(X_not_book, X_book, pairs):
    sum = 0
    book_rev = []
    not_book_rev = []
    book_perc = []
    total_rev = []
    for i in range(len(pairs)):
        sum += (X_book['adjusted_revenue'][pairs[i][0]]) >= (X_not_book['adjusted_revenue'][pairs[i][1]])
        book_rev.append(X_book['notnorm_revenue'][pairs[i][0]])
        not_book_rev.append(X_not_book['notnorm_revenue'][pairs[i][1]])

        book_perc.append((X_book["notnorm_revenue"][pairs[i][0]] - X_not_book["notnorm_revenue"][pairs[i][1]]))#/X_not_book["notnorm_revenue"][pairs[i][1]])
        total_rev.append((X_book["notnorm_revenue"][pairs[i][0]] - X_not_book["notnorm_revenue"][pairs[i][1]]))
    print(np.median(book_rev), np.median(not_book_rev), np.median(np.sort(book_perc)[0:-1]), np.mean(np.sort(book_perc)[0:-1]))
    plt.hist(np.sort(book_perc)[0:-1], bins=100)
    plt.grid()
    print("The revenue of the book based film was better for a total of %d times out of %d (ratio %.4f)" %(sum, len(pairs), sum/len(pairs)))
    return total_rev, book_perc

def extract_films_quizz(X_not_book, X_book,pairs, total_rev, book_perc, num_films) :
    results = []
    for item in range(num_films):
        i = sorted(range(len(total_rev)), key=lambda k: total_rev[k])[-(item + 1)]  # Get index of the top revenue
        results.append({
            "BOB Title": X_book["title"][pairs[i][0]],
            "NOB Title": X_not_book["title"][pairs[i][1]],
            "Difference": round(book_perc[item], 0),
            "BOB Revenue": round(X_book["notnorm_revenue"][pairs[i][0]], 2),
            "NOB Revenue": round(X_not_book["notnorm_revenue"][pairs[i][1]], 2)
        })

    results_df = pd.DataFrame(results)
    return results_df