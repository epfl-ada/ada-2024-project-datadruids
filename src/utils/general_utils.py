import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

color_palette_1 = {
        **{feature: (
            '#CC6677' if feature == 'book_genre_thriller' else
            '#332288' if feature == 'similarity' else
            '#DDCC77' if feature == 'movie_runtime' else
            '#117733' if feature == 'movie_genre_adventure' else
            '#882255' if feature == 'book_in_series' else
            '#44AA99' if feature == 'movie_year' else
            '#999933' if feature == 'book_3_stars_percentage' else
            '#AA4499' if feature == 'movie_adjusted_budget' else
            '#BBCCEE' if feature == 'movie_popularity' else
            '#88CCEE' if feature == 'movie_vote_count' else
            '#CCDDAA' if feature == 'book_in_series' else
            '#EEEEBB' if feature ==  'movie_genres_count' else
            'grey'  # Default grey for everything else
        ) for feature in [
            'book_3_stars_percentage', 'book_rating_count', 'book_genre_thriller',
            'book_in_series', 'book_review_count', 'book_genre_adventure',
            'book_5_stars_percentage', 'book_normalized_rating', 'book_year',
            'book_sentiment_positive', 'book_sentiment_score', 'similarity',
            'movie_sentiment_positive', 'movie_sentiment_score',
            'movie_adjusted_budget', 'movie_adjusted_revenue', 'movie_runtime',
            'movie_year', 'movie_vote_average', 'movie_vote_count',
            'movie_popularity', 'movie_genre_adventure',
            'sentiment_difference', 'book_5_stars_percentage']}
    }

# Define the color palette
color_palette_lin_reg = {
    # Genres in dark grey except genres_adventure
    **{f'genres_{genre}': ('#882255' if genre == 'Adventure' else 'grey') for genre in [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 
        'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War']},
    # Production Countries in dark grey except production_countries_United States of America
    **{f'production_countries_{country}': ('#CC6677' if country == 'United States of America' else 'grey') for country in [
        'Australia', 'Canada', 'China', 'France', 'Germany', 'India', 'Italy', 'Japan', 'Russia', 
        'Spain', 'United Kingdom', 'United States of America']},
    # Spoken Languages in dark grey
    **{f'spoken_languages_{lang}': 'grey' for lang in [
        'English', 'French', 'German', 'Hindi', 'Italian', 'Japanese', 'Mandarin', 'Russian', 'Spanish']},
    # Others with specific colors
    'const': '#2E2E2E',
    'vote_count': '#88CCEE',
    'runtime': '#2E2E2E',
    'adult': 'grey',
    'popularity': '#117733',
    'movie_year': '#44AA99',
    'adjusted_budget': '#DDCC77',
    'adjusted_revenue': '#999933',
    'based_on_book': '#AA4499',
    'vote_average': '#332288'
}

color_palette_book_reg_lin = {
    **{feature: ('#4477AA' if feature == 'rating_count' else
                 '#EE6677' if feature == 'genre_Thriller' else
                 '#CCBB44' if feature == 'genre_Adventure' else
                 '#AA3377' if feature == 'part_of_series' else
                 '#228833' if feature == 'review_count' else
                 '#66CCEE' if feature == 'three_stars_percentage' else
                 '#999933' if feature == 'five_stars_percentage' else
                 'gray') for feature in [
        'const', 'year', 'avg_rating', 'rating_count', 'review_count', 'length',
        'standardized_rating', 'normalized_rating', 'part_of_series',
        'one_star_percentage', 'two_stars_percentage', 'three_stars_percentage',
        'four_stars_percentage', 'five_stars_percentage', 'genre_count',
        'genre_Adventure', 'genre_Childrens', 'genre_Classics', 'genre_Crime',
        'genre_Cultural', 'genre_Fantasy', 'genre_Fiction', 'genre_Historical',
        'genre_Horror', 'genre_Literature', 'genre_Mystery', 'genre_Romance',
        'genre_Science Fiction', 'genre_Thriller', 'genre_Young Adult']}
}

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


def plot_regression(results,  title, color_palette=color_palette_lin_reg) :
   
    to_include = results.params[results.pvalues < 0.10][1:].sort_values() # get only those with significant pvalues
    fig, ax = plt.subplots(figsize=(5, 6), dpi=100)

    # Scatter plot for feature values with matching colors
    colors = [color_palette.get(feature, "#eb5600") for feature in to_include.index]
    ax.scatter(to_include, range(len(to_include)), color=colors, zorder=2)

    # Add labels and title
    ax.set_yticks(range(len(to_include)), to_include.index)

    ax.set_xlabel("Proportional Effect")
    ax.set_title(title, fontsize=16, fontweight='bold', loc='center')


    for idx, ci in enumerate(results.conf_int().loc[to_include.index].iterrows()):
        ax.hlines(idx, ci[1][0], ci[1][1], color=colors[idx], alpha=0.5, zorder=1, linewidth=3)

    # Add a dashed line at 0
    plt.axline((0, 0), (0, 1), color="grey", linestyle="--")
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
    print("The revenue of the book based film was better for a total of %d times out of %d (ratio %.4f)" %(sum, len(pairs), sum/len(pairs)))
    
    print("The median difference of revenues (Bob - Nob) is %d $ while the mean is %d $" %(np.median(np.sort(book_perc)[0:-1]), np.mean(np.sort(book_perc)[0:-1])))
    plt.hist(np.sort(book_perc)[0:-1], bins=100)
    plt.title("Revenue difference between Bobs and Nobs")
    plt.xlabel("Revenue difference [$]")
    plt.grid()
    print("The revenue of the book based film was better for a total of %d times out of %d (ratio %.4f)" %(sum, len(pairs), sum/len(pairs)))
    return total_rev, book_perc

def boxplot_matching(X_not_book, X_book, pairs):

    df_boxplot = pd.DataFrame()
    df_boxplot['NOB'] = X_not_book["notnorm_revenue"][[pairs[i][1] for i in range(len(pairs))]].reset_index(drop=True)
    df_boxplot["BOB"] = X_book["notnorm_revenue"][np.sort([pairs[i][0] for i in range(len(pairs))])].reset_index(drop=True)

    fig = go.Figure()

    fig.add_trace(go.Box(x=df_boxplot['NOB'], name='Nob', boxmean=True, orientation='h', marker_color = '#44AA99', legendrank=2))
    fig.add_trace(go.Box(x=df_boxplot['BOB'], name='Bob', boxmean=True, orientation='h', marker_color = 'sandybrown' ,legendrank=1))

    fig.update_layout(
        title='Revenue of matched Nobs and Bobs',
        xaxis=dict(
            title='Revenue [$]',
            type='log'
        ),
        yaxis=dict(
            title=''
        )
    )
    fig.update_layout(
            width=None,
            height=None,
            template='plotly_white'
        )

    # Show the figure
    fig.show()
    return

def extract_films_quizz(X_not_book, X_book,pairs, total_rev, book_perc, num_films) :
    results = []
    for item in range(num_films):
        i = total_rev.index(np.sort(total_rev)[-(item + 1)])  # Get index of the top revenue
        results.append({
            "BOB Title": X_book["title"][pairs[i][0]],
            "NOB Title": X_not_book["title"][pairs[i][1]],
            "Difference": round(book_perc[i], 0),
            "BOB Revenue": round(X_book["notnorm_revenue"][pairs[i][0]], 2),
            "NOB Revenue": round(X_not_book["notnorm_revenue"][pairs[i][1]], 2)
        })

    results_df = pd.DataFrame(results)
    return results_df

def create_error_plot_regression(results, text, threshold_p_values=0.05, color_palette=color_palette_lin_reg) :
    

    to_include = results.params[results.pvalues < threshold_p_values][1:].sort_values() # get only those with significant pvalues
    confidence_intervals = results.conf_int().loc[to_include.index]
    ci_values = confidence_intervals.values
    colors = [color_palette.get(feature, "#eb5600") for feature in to_include.index]  

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=to_include.values / 1e6,  # Scale x-axis values to 1e6
        y=to_include.index,
        mode='markers',
        marker=dict(size=10, color=colors),
        name="Proportional Effect",
        hoverinfo="text",
        text=[
            f"Feature: {feature}<br>Proportional Effect: {effect / 1e6:.2f} Millions<br>CI: ({ci[0] / 1e6:.2f} Millions, {ci[1] / 1e6:.2f} Millions)"
            for feature, effect, ci in zip(to_include.index, to_include.values.astype(float), ci_values.astype(float))
        ]
    ))


    for i, (ci, color) in enumerate(zip(ci_values, colors)):
        fig.add_shape(
            type="line",
            x0=ci[0] / 1e6,  # Lower CI bound
            x1=ci[1] / 1e6,  # Upper CI bound
            y0=i,  
            y1=i,  
            line=dict(color=color, width=4, dash='solid')  
        )

    fig.add_vline(
        x=0,
        line=dict(color="red", width=1, dash='dash'),
    )
    fig.update_layout(
        title=dict(
            text=text,  # Bold the title
            x=0.02,  
        ),
        xaxis=dict(
            title="Proportional Effect (in Millions)",
            tickvals=[-40, -20, 0, 20, 40, 60, 80],
        ),

        yaxis=dict(
            title="",
            tickmode="array",
            tickvals=list(range(len(to_include.index))),
            ticktext=list(to_include.index)
        ),
        template="plotly_white",
        width=800,
        height=650
    )

    fig.show()


def create_interactive_bar_chart(feature_importances, text, color_palette=color_palette_lin_reg) :
    # Create interactive bar chart
    fig = px.bar(
        feature_importances.head(10),
        y='Feature',
        x='Importance',
        color='Feature',
        color_discrete_map=color_palette,
    )

    fig.update_layout(
        title=dict(
            text=text,  
            x=0.02, 
        ),
        yaxis=dict(
            title="",
            tickmode="array",
        ),
        template="plotly_white",
        width=800,
        height=650,     
        showlegend=False,

    )

    # Add tooltips for better interactivity
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
    )

    fig.show()


def bar_plot_multiple_adaptations(relevant_movies) :
    multi_adapt_2 = relevant_movies[relevant_movies.groupby('BookTitle').BookTitle.transform(len) >= 2]
    multi_adapt_3 = relevant_movies[relevant_movies.groupby('BookTitle').BookTitle.transform(len) >= 3]
    multi_adapt_4 = relevant_movies[relevant_movies.groupby('BookTitle').BookTitle.transform(len) >= 4]
    multi_adapt_5 = relevant_movies[relevant_movies.groupby('BookTitle').BookTitle.transform(len) >= 5]

    Booktitles_2 = multi_adapt_2.value_counts('BookTitle').index.to_list()
    Booktitles_3 = multi_adapt_3.value_counts('BookTitle').index.to_list()
    Booktitles_4 = multi_adapt_4.value_counts('BookTitle').index.to_list()
    Booktitles_5 = multi_adapt_5.value_counts('BookTitle').index.to_list()

    target_columns = ['Runtime', 'Release Year','Similarity','Budget', 'Box Office Revenue']
    dataframe_columns = ['Runtime', 'Release Year','Similarity','Budget', 'Box Office Revenue', 'BookTitle']

    initial =[[0,0,0,0,0,0]]

    Std_2_films = pd.DataFrame(initial,columns=dataframe_columns)
    Std_3_films = pd.DataFrame(initial,columns=dataframe_columns)
    Std_4_films = pd.DataFrame(initial,columns=dataframe_columns)
    Std_5_films = pd.DataFrame(initial,columns=dataframe_columns)

    for title in Booktitles_2:
        df = multi_adapt_2[multi_adapt_2['BookTitle']==title]
        df = df[target_columns]
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df.sort_values('Box Office Revenue', ascending=False, inplace=True)
        df = df.assign(BookTitle=title)
        Std_2_films = pd.concat([Std_2_films,df])
    Std_2_films.reset_index(drop=True,inplace=True)

    for title in Booktitles_3:
        df = multi_adapt_3[multi_adapt_3['BookTitle']==title]
        df = df[target_columns]
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df.sort_values('Box Office Revenue', ascending=False, inplace=True)
        df = df.assign(BookTitle=title)
        Std_3_films = pd.concat([Std_3_films,df])
    Std_3_films.reset_index(drop=True,inplace=True)

    for title in Booktitles_4:
        df = multi_adapt_4[multi_adapt_4['BookTitle']==title]
        df = df[target_columns]
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df.sort_values('Box Office Revenue', ascending=False, inplace=True)
        df = df.assign(BookTitle=title)
        Std_4_films = pd.concat([Std_4_films,df])
    Std_4_films.reset_index(drop=True,inplace=True)

    for title in Booktitles_5:
        df = multi_adapt_5[multi_adapt_5['BookTitle']==title]
        df = df[target_columns]
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df.sort_values('Box Office Revenue', ascending=False, inplace=True)
        df = df.assign(BookTitle=title)
        Std_5_films = pd.concat([Std_5_films,df])
    Std_5_films.reset_index(drop=True,inplace=True)

    Positive_revenue_films_2 = Std_2_films[(Std_2_films['Box Office Revenue']>1.01) | ((Std_2_films['Box Office Revenue'] >0)&(Std_2_films['Box Office Revenue']<0.999 ))]
    Positive_revenue_films_3 = Std_3_films[(Std_3_films['Box Office Revenue']>0) & (Std_3_films['Box Office Revenue'] !=1.000000)]
    Positive_revenue_films_4 = Std_4_films[(Std_4_films['Box Office Revenue']>0) & (Std_4_films['Box Office Revenue'] !=1.000000)]
    Positive_revenue_films_5 = Std_5_films[(Std_5_films['Box Office Revenue']>0) & (Std_5_films['Box Office Revenue'] !=1.000000)]

    colors = {'Runtime':'#2E2E2E',
        'Release Year':'#44AA99',
        'Similarity': 'sandybrown',
        'Budget':'#DDCC77',
        'Box Office Revenue':'#999933'
    }

    datasets = [
        {col: Positive_revenue_films_2[col].values for col in target_columns},
        {col: Positive_revenue_films_3[col].values for col in target_columns},
        {col: Positive_revenue_films_4[col].values for col in target_columns},
        {col: Positive_revenue_films_5[col].values for col in target_columns},
    ]

    # Step 3: Create the initial figure with box plots for the first dataset
    fig = go.Figure()

    for col in target_columns:
        fig.add_trace(
            go.Box(
                y=datasets[0][col],  # Use the first dataset
                name=col,            # Label the boxplot with the column name
                marker_color = colors[col]
                
        )
        )

    # Step 4: Define the slider steps
    steps = []
    for i, dataset in enumerate(datasets):
        step = dict(
            method="update",
            args=[
                {"y": [dataset[col] for col in target_columns]},  # Update y-values for all traces
                {"title": "Standardised features of best performing adaptations"}         # Update the plot title
            ],
            label=f"Bobs with {i + 1} sibling(s) or more"  # Label for the slider step
        )
        steps.append(step)

    # Step 5: Add slider to the layout
    sliders = [dict(
        active=0,
        currentvalue={"prefix": ""},
        pad={"t": 50},  # Padding to position the slider
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title="Standardised features of best performing adaptations",
        yaxis_title="Relative placement in std deviations"
    )


    fig.update_layout(
            width=None,
            height=None,
            template='plotly_white'
        )
    # Display the plot
    fig.show()


def prepare_dataset_for_revbudfrac_approach(regression_dataset_processed_rb_ration_df):

    #Instead of predicting Revenue, the y value will be Revenue divided by Budget
    # ensure that log is defined and we do not divide by 0 later
    regression_dataset_processed_rb_ration_df['adjusted_revenue'] = regression_dataset_processed_rb_ration_df['adjusted_revenue'].clip(lower=1e-10)
    regression_dataset_processed_rb_ration_df['adjusted_budget'] = regression_dataset_processed_rb_ration_df['adjusted_budget'].clip(lower=1e-10)

    regression_dataset_processed_rb_ration_df.loc[regression_dataset_processed_rb_ration_df['adjusted_budget'] == 1, 'adjusted_budget'] = 1 + 1e-10

    regression_dataset_processed_rb_ration_df['log_rb_ratio'] = np.log(regression_dataset_processed_rb_ration_df['adjusted_revenue']) / np.log(regression_dataset_processed_rb_ration_df['adjusted_budget'])
    print("min ratio:", regression_dataset_processed_rb_ration_df['log_rb_ratio'].min())
    print("max ratio:", regression_dataset_processed_rb_ration_df['log_rb_ratio'].max())

    regression_dataset_processed_rb_ration_df.drop(columns=['adjusted_revenue', 'adjusted_budget'], inplace=True)

    return regression_dataset_processed_rb_ration_df

def perform_linear_regression(X_train, X_test, y_train, y_test) :
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    y_pred = results.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    print("RMSE value:", rmse)

    return results