
# coding: utf-8

# # Setup

# In[22]:

import os
import numpy as np
import pandas as pd
import scipy as sp
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)


# In[2]:

start_date = datetime.strptime('2015-01-01', '%Y-%m-%d')
end_date = datetime.now()


# # Data Retrieval

# In[3]:

base_polo_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period=14400'
def get_crypto_data(poloniex_pair):
    '''Retrieve cryptocurrency data from Poloniex'''
    json_url = base_polo_url.format(poloniex_pair, start_date.timestamp(), end_date.timestamp())
    data_df = pd.read_json(json_url)
    data_df = data_df.set_index('date')
    return data_df


# In[4]:

btc_usd_price = get_crypto_data('USDT_BTC')
eth_usd_price = get_crypto_data('USDT_ETH')
ltc_usd_price = get_crypto_data('USDT_LTC')
xrp_usd_price = get_crypto_data('USDT_XRP')
etc_usd_price = get_crypto_data('USDT_ETC')


# In[5]:

#btc_usd_price.head().to_html().replace('\n','')


# In[6]:

#btc_usd_price.describe().to_html().replace('\n','')


# In[7]:

btc_trace = go.Scatter(x=btc_usd_price.index, y=btc_usd_price['weightedAverage'])
py.iplot([btc_trace])


# In[8]:

def compare_many_series(series_arr, label_arr, seperate_y_axis=False, y_axis_label='', scale='linear', initial_hide=False):
    '''Compare multiple series on a scatter plot'''
    layout = go.Layout(
        xaxis=dict(type='date'),
        yaxis=dict(
            title=y_axis_label,
            showticklabels= not seperate_y_axis,
            type=scale
        )
    )
    
    y_axis_config = dict(
        overlaying='y',
        showticklabels=False,
        type=scale )
    
    visibility = 'visible'
    if initial_hide:
        visibility = 'legendonly'
        
    # Form Trace For Each Series
    trace_arr = []
    for index, series in enumerate(series_arr):
        trace = go.Scatter(
            x=series.index, 
            y=series, 
            name=label_arr[index],
            visible=visibility
        )
        
        # Add seperate axis for the series
        if seperate_y_axis:
            trace['yaxis'] = 'y{}'.format(index + 1)
            layout['yaxis{}'.format(index + 1)] = y_axis_config    
        trace_arr.append(trace)

    fig = go.Figure(data=trace_arr, layout=layout)
    py.iplot(fig)


# In[9]:

def df_scatter(df, seperate_y_axis=False, y_axis_label='', scale='linear', initial_hide=False):
    '''Plot a scatter plot for the entire dataframe'''
    data = []
    labels = []
    for col in df.columns:
        data.append(df[col])
        labels.append(col)
    compare_many_series(data, labels, seperate_y_axis, y_axis_label, scale, initial_hide)


# In[10]:

def merge_dfs_on_column(dataframes, labels, col):
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_arr = []
    for df in dataframes:
        series_arr.append(df[col])
    
    series_dict = {}
    for index in range(len(series_arr)):
        series_dict[labels[index]] = series_arr[index]
        
    return pd.DataFrame(series_dict)


# In[11]:

# Combine datasets into single dataframe
currency_dataframes = [btc_usd_price, eth_usd_price, ltc_usd_price, xrp_usd_price, etc_usd_price]
currency_labels = ['BTC', 'ETH', 'LTC', 'XRP', 'ETC']
combined_df = merge_dfs_on_column(currency_dataframes, currency_labels, 'weightedAverage')


# In[12]:

# Filter dataframe to only include datepoints in 2016 or after
combined_df = combined_df[combined_df.index.year >= 2016]


# In[13]:

combined_df.describe()


# In[14]:

trace_1 = go.Scatter(x=combined_df.index, y=combined_df['BTC'], name='BTC VALUE (USD)')
trace_2 = go.Scatter(x=combined_df.index, y=combined_df['ETH'], name='ETH VALUE (USD)')
layout = go.Layout(title='Exchange Rate of BTC and ETH in USD')
fig = go.Figure(data=[trace_1, trace_2], layout=layout)
py.iplot(fig)


# In[15]:

series_0 = btc_usd_price['weightedAverage']
label_0 = 'BTC VALUE (USD)'

series_1 = eth_usd_price['weightedAverage']
label_1 = 'ETH VALUE (USD)'

trace_1 = go.Scatter(x=series_0.index, y=series_0, name=label_0)
trace_2 = go.Scatter(x=series_1.index, y=series_1, name=label_1)
trace_2['yaxis'] = 'y2'

layout = go.Layout(
    title='Exchange Rate of BTC and ETH in USD',
    legend=dict(orientation='h')
)

orange = '#ff7f0e'
blue = '#1f77b4'
scale = 'log'

layout['yaxis1'] = dict(
    title=label_0,
    titlefont=dict(color=blue),
    tickfont=dict(color=blue),
    type=scale
)

layout['yaxis2'] = dict(
    title=label_1,
    overlaying='y',
    titlefont=dict(color=orange),
    tickfont=dict(color=orange),
    side='right',
    type=scale
)


fig = go.Figure(data=[trace_1, trace_2], layout=layout)
py.iplot(fig)


# In[16]:

df_scatter(combined_df, seperate_y_axis=False, y_axis_label='Coin Value (USD)', scale='log', initial_hide=False)


# In[17]:

combined_df_2016 = combined_df[combined_df.index.year == 2016]
combined_df_2016.corr(method='pearson')


# In[21]:

correlation_heatmap(combined_df_2016)


# In[18]:

combined_df_2017 = combined_df[combined_df.index.year == 2017]
combined_df_2017.corr(method='pearson')


# In[19]:

def correlation_heatmap(df, absolute_bounds=True):
    '''Plot a correlation heatmap for the entire dataframe'''
    heatmap = go.Heatmap(
        z=df.corr(method='pearson').as_matrix(),
        x=df.columns,
        y=df.columns,
        colorbar=dict(title='Pearson Coefficient'),
    )
    
    if absolute_bounds:
        heatmap['zmax'] = 1.0
        heatmap['zmin'] = -1.0
    py.iplot([heatmap])


# In[20]:

correlation_heatmap(combined_df_2017)


# In[ ]:



