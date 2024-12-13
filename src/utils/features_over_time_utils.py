import pandas as pd
import plotly.express as px

import pandas as pd
import plotly.express as px


def plot_box_with_trends_over_time(input_df, value_column, title, yaxis_label, use_log_y=False, use_log_x=False,
                                   yaxis_range=None):
    """
    Create a box plot with median, mean trends, and linear regression for a given value column.

    Parameters:
    - input_df: pandas DataFrame with the data.
    - value_column: str, the column name to plot.
    - title: str, the title of the plot.
    - yaxis_label: str, label for the y-axis.
    - use_log_y: bool, if True, use log scale for the y-axis.
    - use_log_x: bool, if True, use log scale for the x-axis.
    - yaxis_range: list or tuple of two values, the lower and upper y-axis limits for default zoom. If None, auto-range.
    """
    df = input_df.copy(deep=True)
    line_colors = {'median_color': 'red', 'mean_color': 'blue', 'regression_color': 'green'}

    df.loc[:, value_column] = pd.to_numeric(df[value_column], errors='coerce')
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df = df.dropna(subset=[value_column, 'year'])
    df['year_group'] = (df['year'] // 3) * 3

    trend = df.groupby('year_group')[value_column].median().reset_index()
    mean_trend = df.groupby('year_group')[value_column].mean().reset_index()

    fig = px.box(df, x='year_group', y=value_column,
                 title=title,
                 labels={'year_group': 'Year Group', value_column: yaxis_label},
                 points=None)

    median_color = line_colors.get('median_color', 'red') if line_colors else 'red'
    fig.add_scatter(x=trend['year_group'], y=trend[value_column], mode='lines+markers',
                    name='Median Trend', line=dict(color=median_color, width=2))

    mean_color = line_colors.get('mean_color', 'blue') if line_colors else 'blue'
    fig.add_scatter(x=mean_trend['year_group'], y=mean_trend[value_column], mode='lines+markers',
                    name='Mean Trend', line=dict(color=mean_color, width=2, dash='dash'))

    fig_regression = px.scatter(trend, x='year_group', y=value_column, trendline='ols')
    regression_line = fig_regression.data[1]
    regression_color = line_colors.get('regression_color', 'green') if line_colors else 'green'
    regression_line.name = 'Linear Regression Median'
    regression_line.line.color = regression_color
    regression_line.showlegend = True
    fig.add_trace(regression_line)

    fig.update_layout(
        xaxis_title='Year Group',
        yaxis_title=yaxis_label,
        width=1000,
        height=600,
    )

    # Set the default zoom range if specified
    if yaxis_range is not None:
        fig.update_layout(
            yaxis=dict(range=yaxis_range)
        )

    # Apply log scale if needed
    if use_log_y:
        fig.update_yaxes(type="log")

    if use_log_x:
        fig.update_xaxes(type="log")

    fig.show()


def plot_movie_count_over_time(input_df, release_date_column, title):
    """
    Create a bar plot showing the number of movies based on books per 3-year interval.

    Parameters:
    - df: pandas DataFrame containing the data.
    - release_date_column: str, the column name for the release date.
    - title: str, the title of the plot.
    """
    df = input_df.copy(deep=True)
    df.loc[:, 'year'] = pd.to_datetime(df[release_date_column], errors='coerce').dt.year
    df = df.dropna(subset=['year'])

    df['year_group'] = (df['year'] // 3) * 3

    trend = df.groupby('year_group').size().reset_index()

    fig = px.bar(trend, x='year_group', y=0,
                 title=title,
                 labels={'year_group': 'Year Group', '0': 'Number of Movies'},
                 text=0)

    fig.update_layout(
        xaxis_title='Year Group',
        yaxis_title='Number of Movies',
        width=1000,
        height=600
    )

    fig.show()
