
# coding: utf-8

# ## A Data-Driven Approach To Cryptocurrency Speculation
# 
# *How do Bitcoin markets behave? What are the causes of the sudden spikes and dips in cryptocurrency values?  Are the markets for different altcoins, such as Litecoin and Ripple, inseparably linked or largely independent?  **How can we predict what will happen next?***
# 
# Articles on cryptocurrencies, such as Bitcoin and Ethereum, are rife with speculation these days, with hundreds of self-proclaimed experts advocating for the trends that they expect to emerge.  What is lacking from many of these analyses is a strong data analysis foundation to backup the claims. 
# 
# The goal of this article is to provide an easy introduction to cryptocurrency analysis using Python.  We will walk through a simple Python script to retrieve, analyze, and visualize data on different cryptocurrencies.  In the process, we will uncover an interesting trend in how these volatile markets behave, and how they are evolving.
# 
# <img id="altcoin_prices_combined_0" src="https://cdn.patricktriest.com/blog/images/posts/crypto-markets/plot-images/altcoin_prices_combined.png" alt="Combined Altcoin Prices">
# 
# This is not a post explaining what cryptocurrencies are (if you want one, I would recommend <a href="https://medium.com/tradecraft-traction/blockchain-for-the-rest-of-us-c3fc5e42254f" target="_blank" rel="noopener">this great overview</a>), nor is it an opinion piece on which specific currencies will rise and which will fall.  Instead, all that we are concerned about in this tutorial is procuring the raw data and uncovering the stories hidden in the numbers.
# 
# 
# ### Step 1 - Setup Your Data Laboratory
# The tutorial is intended to be accessible for enthusiasts, engineers, and data scientists at all skill levels.  The only skills that you will need are a basic understanding of Python and enough knowledge of the command line  to setup a project.
# 
# ##### Step 1.1 - Install Anaconda
# The easiest way to install the dependencies for this project from scratch is to use Anaconda, a prepackaged Python data science ecosystem and dependency manager.
# 
# To setup Anaconda, I would recommend following the official installation instructions - [https://www.continuum.io/downloads](https://www.continuum.io/downloads).  
# 
# *If you're an advanced user, and you don't want to use Anaconda, that's totally fine; I'll assume you don't need help installing the required dependencies.  Feel free to skip to section 2.*
# 
# ##### Step 1.2 - Setup an Anaconda Project Environment
# 
# Once Anaconda is installed, we'll want to create a new environment to keep our dependencies organized.
# 
# Run `conda create --name cryptocurrency-analysis python=3` to create a new Anaconda environment for our project.
# 
# Next, run `source activate cryptocurrency-analysis` (on Linux/macOS) or `activate cryptocurrency-analysis` (on windows) to activate this environment.
# 
# Finally, run `conda install numpy pandas nb_conda jupyter plotly quandl` to install the required dependencies in the environment.  This could take a few minutes to complete.
# 
# *Why use environments?  If you plan on developing multiple Python projects on your computer, it is helpful to keep the dependencies (software libraries and packages) separate in order to avoid conflicts.  Anaconda will create a special environment directory for the dependencies for each project to keep everything organized and separated.*
# 
# ##### Step 1.3 - Start An Interative Jupyter Notebook
# 
# Once the environment and dependencies are all set up, run `jupyter notebook` to start the iPython kernel, and open your browser to `http://localhost:8888/`.  Create a new Python notebook, making sure to use the `Python [conda env:cryptocurrency-analysis]` kernel.
# 
# ![Empty Jupyer Notebook](https://cdn.patricktriest.com/blog/images/posts/crypto-markets/jupyter-setup.png)
# 

# #####  Step 1.4 - Import the Dependencies At The Top of The Notebook
# Once you've got a blank Jupyter notebook open, the first thing we'll do is import the required dependencies.

# In[1]:


import os
import numpy as np
import pandas as pd
import pickle
import quandl
from datetime import datetime


# We'll also import Plotly and enable the offline mode.

# In[2]:


import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)


# In[3]:


quandl.ApiConfig.api_key = os.environ['QUANDL_API_KEY']


# ### Step 2 - Retrieve Bitcoin Pricing Data
# Now that everything is set up, we're ready to start retrieving data for analysis.  First, we need to get Bitcoin pricing data using [Quandl's free Bitcoin API](https://blog.quandl.com/api-for-bitcoin-data).

# ##### Step 2.1 - Define Quandl Helper Function
# To assist with this data retrieval we'll define a function to download and cache datasets from Quandl.

# In[4]:


def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df


# We're using `pickle` to serialize and save the downloaded data as a file, which will prevent our script from re-downloading the same data each time we run the script.  The function will return the data as a [Pandas]('http://pandas.pydata.org/') dataframe.  If you're not familiar with dataframes, you can think of them as super-powered Python spreadsheets.

# ##### Step 2.2 - Pull Kraken Exchange Pricing Data
# Let's first pull the historical Bitcoin exchange rate for the [Kraken](https://www.kraken.com/) Bitcoin exchange.

# In[5]:


# Pull Kraken BTC price exchange data
btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD')


# We can inspect the first 5 rows of the dataframe using the `head()` method.

# In[6]:


btc_usd_price_kraken.head()


# Next, we'll generate a simple chart as a quick visual verification that the data looks correct.  

# In[7]:


# Chart the BTC pricing data
btc_trace = go.Scatter(x=btc_usd_price_kraken.index, y=btc_usd_price_kraken['Weighted Price'])
py.iplot([btc_trace])


# Here, we're using [Plotly](https://plot.ly/) for generating our visualizations.  This is a less traditional choice than some of the more established Python data visualization libraries such as [Matplotlib](https://matplotlib.org/), but I think Plotly is a great choice since it produces fully-interactive charts using [D3.js](https://d3js.org/).  These charts have attractive visual defaults, are easy to explore, and are very simple to embed in web pages.
# 
# > As a quick sanity check, you should compare the generated chart with publically available graphs on Bitcoin prices(such as those on [Coinbase](https://www.coinbase.com/dashboard)), to verify that the downloaded data is legit.

# ##### Step 2.3 - Pull Pricing Data From More BTC Exchanges
# You might have noticed a hitch in this dataset - there are a few notable down-spikes, particularly in late 2014 and early 2016.  These spikes are specific to the Kraken dataset, and we obviously don't want them to be reflected in our overall pricing analysis.  
# 
# The nature of Bitcoin exchanges is that the pricing is determined by supply and demand, hence no single exchange contains a true "master price" of Bitcoin.  To solve this issue, along with that of down-spikes, we'll pull data from three more major Bitcoin changes to calculate an aggregate Bitcoin price index.
# 
# First, we will download the data from each exchange into a dictionary of dataframes.

# In[8]:


# Pull pricing data for 3 more BTC exchanges
exchanges = ['COINBASE','BITSTAMP','ITBIT']

exchange_data = {}

exchange_data['KRAKEN'] = btc_usd_price_kraken

for exchange in exchanges:
    exchange_code = 'BCHARTS/{}USD'.format(exchange)
    btc_exchange_df = get_quandl_data(exchange_code)
    exchange_data[exchange] = btc_exchange_df


# ##### Step 2.4 - Merge All Of The Pricing Data Into A Single Dataframe
# Next, we will define a simple function to merge a common column of each dataframe into a new combined dataframe.

# In[9]:


def merge_dfs_on_column(dataframes, labels, col):
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]
        
    return pd.DataFrame(series_dict)


# Now we will merge all of the dataframes together on their "Weighted Price" column.

# In[10]:


# Merge the BTC price dataseries' into a single dataframe
btc_usd_datasets = merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Weighted Price')


# Finally, we can preview last five rows the result using the `tail()` method, to make sure it looks ok.

# In[11]:


btc_usd_datasets.tail()


# ##### Step 2.5 - Visualize The Pricing Datasets
# The next logical step is to visualize how these pricing datasets compare.  For this, we'll define a helper function to provide a single-line command to compare each column in the dataframe on a graph using Plotly.

# In[12]:


def df_scatter(df, title, seperate_y_axis=False, y_axis_label='', scale='linear', initial_hide=False):
    '''Generate a scatter plot of the entire dataframe'''
    label_arr = list(df)
    series_arr = list(map(lambda col: df[col], label_arr))
    
    layout = go.Layout(
        title=title,
        legend=dict(orientation="h"),
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


# In the interest of brevity, I won't go too far into how this helper function works.  Check out the documentation for [Pandas](http://pandas.pydata.org/) and [Plotly](https://plot.ly/) if you would like to learn more. 

# With the function defined, we can compare our pricing datasets like so.

# In[13]:


# Plot all of the BTC exchange prices
df_scatter(btc_usd_datasets, 'Bitcoin Price (USD) By Exchange')


# ##### Step 2.6 - Clean and Aggregate the Pricing Data
# We can see that, although the four series follow roughly the same path, there are various irregularities in each that we'll want to get rid of.
# 
# Let's remove all of the zero values from the dataframe, since we know that the price of Bitcoin has never been equal to zero in the timeframe that we are examining.

# In[14]:


# Remove "0" values
btc_usd_datasets.replace(0, np.nan, inplace=True)


# When we re-chart the dataframe, we'll see a much cleaner looking chart without the spikes.

# In[15]:


# Plot the revised dataframe
df_scatter(btc_usd_datasets, 'Bitcoin Price (USD) By Exchange')


# We can now calculate a new column, containing the daily average Bitcoin price across all of the exchanges.

# In[16]:


# Calculate the average BTC price as a new column
btc_usd_datasets['avg_btc_price_usd'] = btc_usd_datasets.mean(axis=1)


# This new column is our Bitcoin pricing index!  Let's chart that column to make sure it looks ok.

# In[17]:


# Plot the average BTC price
btc_trace = go.Scatter(x=btc_usd_datasets.index, y=btc_usd_datasets['avg_btc_price_usd'])
py.iplot([btc_trace])


# Yup, looks good.  We'll use this aggregate pricing series later on, in order to convert the exchange rates of other cryptocurrencies to USD.

# ### Step 3 - Retrieve Altcoin Pricing Data
# Now that we have a solid time series dataset for the price of Bitcoin, let's pull in some data on non-Bitcoin cryptocurrencies, commonly referred to as altcoins.

# ##### Step 3.1 - Define Poloniex API Helper Functions
# 
# For retrieving data on cryptocurrencies we'll be using the [Poloniex API](https://poloniex.com/support/api/).  To assist in the altcoin data retrieval, we'll define two helper functions to download and cache JSON data from this API.
# 
# First, we'll define `get_json_data`, which will download and cache JSON data from a provided URL.

# In[18]:


def get_json_data(json_url, cache_path):
    '''Download and cache JSON data, return as a dataframe.'''
    try:        
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(json_url))
    except (OSError, IOError) as e:
        print('Downloading {}'.format(json_url))
        df = pd.read_json(json_url)
        df.to_pickle(cache_path)
        print('Cached response at {}'.format(json_url, cache_path))
    return df


# Next, we'll define a function to format Poloniex API HTTP requests and call our new `get_json_data` function to save the resulting data.

# In[19]:


base_polo_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'
start_date = datetime.strptime('2015-01-01', '%Y-%m-%d') # get data from the start of 2015
end_date = datetime.now() # up until today
pediod = 86400 # pull daily data (86,400 seconds per day)

def get_crypto_data(poloniex_pair):
    '''Retrieve cryptocurrency data from poloniex'''
    json_url = base_polo_url.format(poloniex_pair, start_date.timestamp(), end_date.timestamp(), pediod)
    data_df = get_json_data(json_url, poloniex_pair)
    data_df = data_df.set_index('date')
    return data_df


# This function will take a cryptocurrency pair string (such as 'BTC_ETH') and return the dataframe containing the historical exchange rate of the two currencies.

# ##### Step 3.2 - Download Trading Data From Poloniex
# Most altcoins cannot be bought directly with USD; to acquire these coins individuals often buy Bitcoins and then trade the Bitcoins for altcoins on cryptocurrency exchanges.  For this reason we'll be downloading the exchange rate to BTC for each coin, and then we'll use our existing BTC pricing data to convert this value to USD.

# We'll download exchange data for nine of the top cryptocurrencies -
# [Ethereum](https://www.ethereum.org/), [Litecoin](https://litecoin.org/), [Ripple](https://ripple.com/), [Ethereum Classic](https://ethereumclassic.github.io/), [Stellar](https://www.stellar.org/), [Dashcoin](http://dashcoin.info/), [Siacoin](http://sia.tech/), [Monero](https://getmonero.org/), and [NEM](https://www.nem.io/).

# In[20]:


altcoins = ['ETH','LTC','XRP','ETC','STR','DASH','SC','XMR','XEM']

altcoin_data = {}
for altcoin in altcoins:
    coinpair = 'BTC_{}'.format(altcoin)
    crypto_price_df = get_crypto_data(coinpair)
    altcoin_data[altcoin] = crypto_price_df


# Now we have a dictionary of 9 dataframes, each containing the historical daily average exchange prices between the altcoin and Bitcoin.
# 
# We can preview the last few rows of the Ethereum price table to make sure it looks ok.

# In[21]:


altcoin_data['ETH'].tail()


# ##### Step 3.3 - Convert Prices to USD
# 
# Since we now have the exchange rate for each cryptocurrency to Bitcoin, and we have the Bitcoin/USD historical pricing index, we can directly calculate the USD price series for each altcoin.

# In[22]:


# Calculate USD Price as a new column in each altcoin dataframe
for altcoin in altcoin_data.keys():
    altcoin_data[altcoin]['price_usd'] =  altcoin_data[altcoin]['weightedAverage'] * btc_usd_datasets['avg_btc_price_usd']


# Here, we've created a new column in each altcoin dataframe with the USD prices for that coin.
# 
# Next, we can re-use our `merge_dfs_on_column` function from earlier to create a combined dataframe of the USD price for each cryptocurrency.

# In[23]:


# Merge USD price of each altcoin into single dataframe 
combined_df = merge_dfs_on_column(list(altcoin_data.values()), list(altcoin_data.keys()), 'price_usd')


# Easy.  Now let's also add the Bitcoin prices as a final column to the combined dataframe.

# In[24]:


# Add BTC price to the dataframe
combined_df['BTC'] = btc_usd_datasets['avg_btc_price_usd']


# Now we should have a single dataframe containing daily USD prices for the ten cryptocurrencies that we're examining.
# 
# Let's reuse our `df_scatter` function from earlier to chart all of the cryptocurrency prices against each other.

# In[25]:


# Chart all of the altocoin prices
df_scatter(combined_df, 'Cryptocurrency Prices (USD)', seperate_y_axis=False, y_axis_label='Coin Value (USD)', scale='log')


# Nice! This graph gives a pretty solid "big picture" view of how the exchange rates of each currency have varied over the past few years.  
# 
# > Note that we're using a logarithmic y-axis scale in order to compare all of currencies on the same plot.  You are welcome to try out different parameters values here (such as `scale='linear'`) to get different perspectives on the data.

# ##### Step 3.4 - Compute Correlation Values of The Cryptocurrencies
# You might notice is that the cryptocurrency exchange rates, despite their wildly different values and volatility, seem to be slightly correlated. Especially since the spike in April 2017, even many of the smaller fluctuations appear to be occurring in sync across the entire market.  
# 
# A visually-derived hunch is not much better than a guess until we have the stats to back it up.
# 
# We can test our correlation hypothesis using the Pandas `corr()` method, which computes a Pearson correlation coefficient for each column in the dataframe against each other column.
# 
# Computing correlations directly on a non-stationary time series (such as raw pricing data) can give biased correlation values.  We will work around this by using the `pct_change()` method, which will convert each cell in the dataframe from an absolute price value to a daily return percentage.
# 
# First we'll calculate correlations for 2016.

# In[26]:


# Calculate the pearson correlation coefficients for altcoins in 2016
combined_df_2016 = combined_df[combined_df.index.year == 2016]
combined_df_2016.pct_change().corr(method='pearson')


# These correlation coefficients are all over the place.  Coefficients close to 1 or -1 mean that the series' are strongly correlated or inversely correlated respectively, and coefficients close to zero mean that the values tend to fluctuate independently of each other.
# 
# To help visualize these results, we'll create one more helper visualization function.

# In[27]:


def correlation_heatmap(df, title, absolute_bounds=True):
    '''Plot a correlation heatmap for the entire dataframe'''
    heatmap = go.Heatmap(
        z=df.corr(method='pearson').as_matrix(),
        x=df.columns,
        y=df.columns,
        colorbar=dict(title='Pearson Coefficient'),
    )
    
    layout = go.Layout(title=title)
    
    if absolute_bounds:
        heatmap['zmax'] = 1.0
        heatmap['zmin'] = -1.0
        
    fig = go.Figure(data=[heatmap], layout=layout)
    py.iplot(fig)


# In[28]:


correlation_heatmap(combined_df_2016.pct_change(), "Cryptocurrency Correlations in 2016")


# Here, the dark red values represent strong correlations (note that each currency is, obviously, strongly correlated with itself), and the dark blue values represent strong inverse correlations.  All of the light blue/orange/gray/tan colors in-between represent varying degrees of weak/non-existent correlations.
# 
# What does this chart tell us? Essentially, it shows that there was very little statistically significant linkage between how the prices of different cryptocurrencies fluctuated during 2016.
# 
# Now, to test our hypothesis that the cryptocurrencies have become more correlated in recent months, let's repeat the same test using only the data from 2017.

# In[29]:


combined_df_2017 = combined_df[combined_df.index.year == 2017]
combined_df_2017.pct_change().corr(method='pearson')


# These are somewhat more significant correlation coefficients.  Strong enough to use as the sole basis for an investment? Certainly not.  
# 
# It is notable, however, that almost all of the cryptocurrencies have become more correlated with each other across the board.

# In[30]:


correlation_heatmap(combined_df_2017.pct_change(), "Cryptocurrency Correlations in 2017")


# Huh. That's rather interesting.

# ### Why is this happening?
# 
# Good question.  I'm really not sure.  
# 
# The most immediate explanation that comes to mind is that **hedge funds have recently begun publicly trading in crypto-currency markets**[^1][^2].  These funds have vastly more capital to play with than the average trader, so if a fund is hedging their bets across multiple cryptocurrencies, and using similar trading strategies for each based on independent variables (say, the stock market), it could make sense that this trend would emerge.
# 
# ##### In-Depth - XRP and STR
# For instance, one noticeable trait of the above chart is that XRP (the token for [Ripple](https://ripple.com/)), is the least correlated cryptocurrency.  The notable exception here is with STR (the token for [Stellar](https://www.stellar.org/), officially known as "Lumens"), which has a stronger (0.62) correlation with XRP.  
# 
# What is interesting here is that Stellar and Ripple are both fairly similar fintech platforms aimed at reducing the friction of international money transfers between banks.  
# 
# It is conceivable that some big-money players and hedge funds might be using similar trading strategies for their investments in Stellar and Ripple, due to the similarity of the blockchain services that use each token. This could explain why XRP is so much more heavily correlated with STR than with the other cryptocurrencies.
# 
# > Quick Plug - I'm a contributor to [Chipper](https://www.chipper.xyz/), a (very) early-stage startup using Stellar with the aim of disrupting micro-remittances in Africa.
# 
# ### Your Turn
# 
# This explanation is, however, largely speculative.  **Maybe you can do better**.  With the foundation we've made here, there are hundreds of different paths to take to continue searching for stories within the data.  
# 
# Here are some ideas:
# 
# - Add data from more cryptocurrencies to the analysis.
# - Adjust the time frame and granularity of the correlation analysis, for a more fine or coarse grained view of the trends. 
# - Search for trends in trading volume and/or blockchain mining data sets.  The buy/sell volume ratios are likely more relevant than the raw price data if you want to predict future price fluctuations.
# - Add pricing data on stocks, commodities, and fiat currencies to determine which of them correlate with cryptocurrencies (but please remember the old adage that "Correlation does not imply causation").
# - Quantify the amount of "buzz" surrounding specific cryptocurrencies using Event Registry, GDLELT, and Google Trends. 
# - Train a predictive machine learning model on the data to predict tomorrow's prices.  If you're more ambitious, you could even try doing this with a recurrent neural network (RNN).
# - Use your analysis to create an automated "Trading Bot" on a trading site such as Poloniex or Coinbase, using their respective trading APIs.  Be careful: a poorly optimized trading bot is an easy way to lose your money quickly.
# - **Share your findings!**  The best part of Bitcoin, and of cryptocurrencies in general, is that their decentralized nature makes them more free and democratic than virtually any other market.  Open source your analysis, participate in the community, maybe write a blog post about it.
# 
# Hopefully, now you have the skills to do your own analysis and to think critically about any speculative cryptocurrency articles you might read in the near future, especially those written without any data to back up the provided predictions.
# 
# Thanks for reading, and feel free to comment below with any ideas, suggestions, or criticisms regarding this tutorial.  I've got second (and potentially third) part in the works, which will likely be following through on some of same the ideas listed above, so stay tuned for more in the coming weeks.
# 
# [^1]: http://fortune.com/2017/07/26/bitcoin-cryptocurrency-hedge-fund-sequoia-andreessen-horowitz-metastable/
# [^2]: https://www.forbes.com/sites/laurashin/2017/07/12/crypto-boom-15-new-hedge-funds-want-in-on-84000-returns/#7946ab0d416a
