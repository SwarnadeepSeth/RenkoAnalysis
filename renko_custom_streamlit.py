import streamlit as st
import yfinance as yf
import pandas as pd
from stocktrends import Renko
import mplfinance as mpf
import datetime, os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Wide mode for better layout
st.set_page_config(layout="wide")

# Custom mpl plot style definition
custom_style = mpf.make_mpf_style(
    base_mpf_style='yahoo',
    rc={
        'figure.facecolor': '#f0f0f0',
        'axes.facecolor': '#ffffff',
        'axes.edgecolor': '#000000',
        'grid.color': '#dddddd',
        'font.size': 12,
        'axes.labelsize': 20,
        'axes.titlesize': 20
    },
    y_on_right=False,
    gridstyle='--',
    gridaxis='both'
)

# Define a function to clean the data
def clean_data(data):
    # Reset index to ensure 'Date' is a column
    data = data.reset_index()

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(filter(None, col)).strip() for col in data.columns.values]

    # Remove ticker suffixes like '_INDIGO.NS'
    data.columns = [col.split('_')[0] if '_' in col else col for col in data.columns]

    # Convert required columns to numeric
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    return data.reset_index(drop=True)

# Fetch data from Yahoo Finance
@st.cache_data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return clean_data(data)

# Calculate Renko bricks
def calculate_renko(df, percentage_brick_size):
    df2 = df.copy()
    df2.columns = [col.lower() for col in df2.columns]
    renko = Renko(df2)
    renko.brick_size = max(0.5, round(0.01 * percentage_brick_size* df2['close'].iloc[-1], 2))
    print (renko.brick_size)
    renko_df = renko.get_ohlc_data()

    # Convert to float
    for col in ['open', 'high', 'low', 'close']:
        renko_df[col] = pd.to_numeric(renko_df[col], errors='coerce')
    
    renko_df['Date'] = pd.to_datetime(renko_df['date'])
    renko_df.set_index('Date', inplace=True)

    renko_df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close'
    }, inplace=True)

    return renko_df

def find_last_peak(renko_df):
    trend_series = renko_df['Close']
    peaks, properties = find_peaks(trend_series, prominence=1)
    last_peak_index = peaks[-1]
    last_peak_date = renko_df.index[last_peak_index]
    last_peak_close = trend_series[last_peak_index]
    return last_peak_index, last_peak_date, last_peak_close

def find_last_trough(renko_df):
    trend_series = renko_df['Close']
    troughs, properties = find_peaks(-trend_series, prominence=1)
    last_trough_index = troughs[-1]
    last_trough_date = renko_df.index[last_trough_index]
    last_trough_close = trend_series[last_trough_index]
    return last_trough_index, last_trough_date, last_trough_close

def plot_Renko(ticker, renko_df, last_peak_close):
    # Calculate 26-day EMA
    ema_26 = renko_df['Close'].ewm(span=26, adjust=False).mean()
    
    # Define additional plots
    ap = [
        mpf.make_addplot([last_peak_close] * len(renko_df), type='line', color='xkcd:green', secondary_y=False),
        mpf.make_addplot(ema_26, type='line', color='xkcd:blue')
    ]
    fig, ax = mpf.plot(renko_df, type='candle', style=custom_style,
                    xlabel='Date', ylabel='Price', volume=False, show_nontrading=False, figsize=(14, 5),
                    addplot=ap, tight_layout=False,
                    datetime_format='%d/%m/%y',
                    returnfig=True)  # Return the figure and axes

    plt.setp(ax[0].get_xticklabels(), rotation=0)
    plt.setp(ax[0], title=f'{ticker}')
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

# ==================================================================================================
st.title('Stock Analysis with Renko')

with st.sidebar:
    # Dropdown for stock momentum period
    period = st.selectbox("Select stock momentum period:", ['3months', '6 months', '9 months'])

    # Convert period to appropriate time delta
    period_dict = {'3months': 90, '6 months': 180, '9 months': 270}
    period = period_dict[period]

    # Dropdown for percentage brick size
    percentage_brick_size = st.selectbox("Select percentage brick size:", ['1.0', '3.0', '5.0'])

    # Convert percentage_brick_size to float
    percentage_brick_size = float(percentage_brick_size)

    # Checkbox for IND_stocks
    IND_stocks = st.checkbox("IND_stocks", value=True)

# ==================================================================================================
if st.button('Analyze Stocks'):
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=period)).strftime('%Y-%m-%d')

    if IND_stocks:
        # List of Nifty 200 stocks
        stock_list = pd.read_csv('ind_nifty200list.csv')
        stock_list = stock_list['Symbol'].tolist()

        # Add .NS to all the stock symbols
        stock_list = [stock + '.NS' for stock in stock_list]
    else:
        # List of US stocks
        stock_list = pd.read_csv('nasdaq_list.csv')
        stock_list = stock_list['Symbol'].tolist()[0:200]

    for ticker in stock_list:
        print ("Downloading data for:", ticker)

        try:
            # Fetch data
            df = fetch_data(ticker, start_date, end_date)
            renko_df = calculate_renko(df, percentage_brick_size)

            # Get the peak
            last_peak_index, last_peak_date, last_peak_close = find_last_peak(renko_df)

            # Buy Condition
            if last_peak_close < renko_df['Close'].iloc[-1]:
                print (f"Buy Condition for {ticker}")

                # Plot the Renko chart
                plot_Renko(ticker, renko_df, last_peak_close)
        
        except Exception as e:
            print (e)



