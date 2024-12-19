import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
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