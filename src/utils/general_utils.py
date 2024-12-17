import pandas as pd

def adjust_for_inflation_final_dataset(data, df_revenue) :
    inflation = pd.read_csv('./../data/CPIAUCNS.csv', parse_dates=['DATE'])

    inflation['Year'] = inflation['DATE'].dt.year

    inflation_yearly = inflation.groupby('Year')['CPIAUCNS'].mean().reset_index()
    inflation_yearly.columns = ['Year', 'CPI']

    base_year = data.movie_year.max()
    base_cpi = inflation_yearly[inflation_yearly['Year'] == base_year]['CPI'].values[0]

    df_revenue = df_revenue.merge(inflation_yearly, left_on='movie_year', right_on='Year', how='left')

    df_revenue['AdjustedRevenue'] = df_revenue['revenue'] * (base_cpi / df_revenue['CPI'])
    
    return df_revenue


