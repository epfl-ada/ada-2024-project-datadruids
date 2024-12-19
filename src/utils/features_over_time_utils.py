import pandas as pd
import plotly.express as px
import numpy as np
from scipy.optimize import curve_fit

import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import curve_fit
import plotly.graph_objects as go


def plot_box_with_trends_over_time(input_df, column_name, title, y_axis_label, use_log_y=False, use_log_x=False,
                                   y_axis_range=None, use_exponential_regression=False, should_save_to_html=False):
    """
    Create a box plot with median, mean trends, and linear or exponential regression for a given value column.
    The data is grouped by 3-year intervals.
    """
    data_df = input_df.copy(deep=True)
    line_colors = {'median_color': 'red', 'mean_color': 'blue', 'regression_color': 'green'}

    data_df.loc[:, column_name] = pd.to_numeric(data_df[column_name], errors='coerce')
    data_df['year'] = pd.to_datetime(data_df['release_date'], errors='coerce').dt.year
    data_df = data_df.dropna(subset=[column_name, 'year'])
    data_df['year_group'] = (data_df['year'] // 3) * 3

    trend = data_df.groupby('year_group')[column_name].median().reset_index()
    mean_trend = data_df.groupby('year_group')[column_name].mean().reset_index()

    fig = px.box(data_df, x='year_group', y=column_name,
                 title=title,
                 labels={'year_group': 'Year Group', column_name: y_axis_label},
                 points=None)

    # Make box plot toggleable
    fig.data[0].showlegend = True
    fig.data[0].name = "Box Plots"

    mean_color = line_colors.get('mean_color', 'blue') if line_colors else 'blue'
    fig.add_scatter(x=mean_trend['year_group'], y=mean_trend[column_name], mode='lines+markers',
                    name='Mean Trend', line=dict(color=mean_color, width=2, dash='dash'))

    median_color = line_colors.get('median_color', 'red') if line_colors else 'red'
    fig.add_scatter(x=trend['year_group'], y=trend[column_name], mode='lines+markers',
                    name='Median Trend', line=dict(color=median_color, width=2))

    # Exponential regression
    def exponential_func(x, a, b):
        return a * np.exp(b * x)

    if use_exponential_regression:
        x_data = trend['year_group'].values
        y_data = trend[column_name].values

        valid_mask = y_data > 0 # Otherwise, not possible to do exp. regr.
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]

        initial_a = np.max(y_data)
        initial_b = 0.01
        try:
            # if maxfev is too small, there might not be enough iterations to converge
            popt, _ = curve_fit(exponential_func, x_data, y_data, p0=(initial_a, initial_b), maxfev=10000)
            y_regression = exponential_func(x_data, *popt)

            regression_color = line_colors.get('regression_color', 'green') if line_colors else 'green'
            fig.add_scatter(x=trend['year_group'], y=y_regression, mode='lines',
                            name='Exponential Regression Median',
                            line=dict(color=regression_color, width=2))
        except RuntimeError:
            print("Exp. regr. did not converge...")
    else:
        # Linear regression
        fig_regression = px.scatter(trend, x='year_group', y=column_name, trendline='ols')
        regression_line = fig_regression.data[1]
        regression_color = line_colors.get('regression_color', 'green') if line_colors else 'green'
        regression_line.name = 'Linear Regression Median'
        regression_line.line.color = regression_color
        regression_line.showlegend = True
        fig.add_trace(regression_line)

    fig.update_layout(
        xaxis_title='Year Group',
        yaxis_title=y_axis_label,
        width=None,
        height=None,
    )

    # Potentially adapt figure configurations
    if y_axis_range is not None:
        fig.update_layout(
            yaxis=dict(range=y_axis_range)
        )

    if use_log_y:
        fig.update_yaxes(type="log")

    if use_log_x:
        fig.update_xaxes(type="log")

    if should_save_to_html:

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .plotly-graph-div {{
                    width: 100% !important;
                    height: 100% !important;
                }}
            </style>
        </head>
        <body>
            <div id="plotly-div"></div>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                var plotly_data = {fig.to_json()};
                Plotly.newPlot('plotly-div', plotly_data.data, plotly_data.layout);
            </script>
        </body>
        </html>
        """

        with open(f'{title}.html', 'w') as f:
            f.write(html_content)

    fig.show()
    return fig


def overlay_two_figures(title, figure_1, figure_2, figure_1_type, figure_2_type, y_axis_range=None, should_save_to_html=False, should_display_means=True):
    """
    Combine two figures (generated by plot_box_with_trends_over_time) and overlay their results
    while maintaining toggleability from the legend. The figure_type string is prepended to the legend names.
    """
    overlayed_fig = go.Figure()

    # colors picked with color-blind palette Tol:
    # https://davidmathlogic.com/colorblind/#%23332288-%23117733-%2344AA99-%2388CCEE-%23DDCC77-%23CC6677-%23AA4499-%23882255
    colors_fig1 = {
        'box_color': 'gray',
        'mean_trend_color': '#332288',
        'median_trend_color': '#AA4499',
        'regression_color': '#CC6677'
    }

    colors_fig2 = {
        'box_color': '#DDCC77',
        'mean_trend_color': '#117733',
        'median_trend_color': '#44AA99',
        'regression_color': '#88CCEE'
    }

    # set colors for each trace
    for trace, colors, fig_type in zip(figure_1.data + figure_2.data, [colors_fig1] * len(figure_1.data) + [colors_fig2] * len(figure_2.data), [figure_1_type] * len(figure_1.data) + [figure_2_type] * len(figure_2.data)):
        trace_name = trace.name
        if fig_type not in trace.name:
            trace.name = f"{fig_type} - {trace_name}"

        if 'Box' in trace.name:
            trace.line.color = colors['box_color']
            trace.visible = 'legendonly'
            trace.marker = dict(outliercolor=colors['box_color'])
        elif 'Median Trend' in trace.name:
            trace.line.color = colors['median_trend_color']
        elif 'Mean Trend' in trace.name:
            trace.line.color = colors['mean_trend_color']
            if not should_display_means:
                trace.visible = 'legendonly'          
        elif 'Regression' in trace.name:
            trace.line.color = colors['regression_color']
        overlayed_fig.add_trace(trace)

    overlayed_fig.update_layout(
        title=f"{title} - Overlay",
        xaxis_title=figure_1.layout.xaxis.title,
        yaxis_title=figure_1.layout.yaxis.title,
        width=1000,
        height=600,
    )

    # set default y-axis range if provided
    if y_axis_range is not None:
        overlayed_fig.update_layout(
            yaxis=dict(range=y_axis_range)
        )

    overlayed_fig.show()

    if should_save_to_html:
        overlayed_fig.write_html(f'{title}_overlay.html', full_html=False)

    return overlayed_fig


def plot_movie_count_over_time(input_df, release_date_column, title):
    """
    Create a bar plot showing the number of movies based on books per 3-year interval.
    """
    data_df = input_df.copy(deep=True)
    data_df.loc[:, 'year'] = pd.to_datetime(data_df[release_date_column], errors='coerce').dt.year
    data_df = data_df.dropna(subset=['year'])

    data_df['year_group'] = (data_df['year'] // 3) * 3

    trend = data_df.groupby('year_group').size().reset_index()

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
