import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from pykalman import KalmanFilter
from datetime import datetime, timedelta
import plotly.graph_objs as go

# Set up Streamlit page
st.title('Kalman Filter-Based Stock Trading Strategy')

# User Inputs
st.sidebar.header("User Inputs")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., DJT):", value="DJT")
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=1095))  # Adjusted to 3 years
end_date = st.sidebar.date_input("End Date", datetime.today())
transition_covariance = st.sidebar.number_input("Transition Covariance", value=0.01, step=0.01, min_value=0.0)
observation_covariance = st.sidebar.number_input("Observation Covariance", value=1.0, step=0.1, min_value=0.0)

# Adjustable thresholds for buy and take profit signals
buy_signal_threshold = st.sidebar.number_input("Buy Signal Threshold", value=-0.44, step=0.01)
take_profit_threshold = st.sidebar.number_input("Take Profit Threshold", value=0.25, step=0.01)

# Function to apply the Kalman Filter
def kalman_filter(close_prices):
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )
    state_means, _ = kf.filter(close_prices)
    return state_means.flatten()

# Function to calculate Z-scores
def calculate_z_scores(prices, kalman_avg):
    std_dev = np.std(prices)
    z_scores = (prices - kalman_avg) / std_dev
    return z_scores

# Function to generate signals based on Z-scores
def generate_signals(z_scores, close_prices):
    signals = []
    trades = []
    last_buy_price = None
    last_buy_date = None

    for i in range(1, len(z_scores)):
        curr_z_score = z_scores[i]
        timestamp = stock_data.index[i]

        if last_buy_price is None:  # No previous buy
            if curr_z_score <= buy_signal_threshold:
                last_buy_price = close_prices[i]
                last_buy_date = timestamp
                signals.append('Buy Signal')
            else:
                signals.append('No Action')
        else:  # There is a previous buy
            # Check for take profit only if we have an entry price
            if curr_z_score >= take_profit_threshold:
                take_profit_price = close_prices[i]
                profit_dollars = take_profit_price - last_buy_price
                profit_percent = (profit_dollars / last_buy_price) * 100 if last_buy_price > 0 else 0
                duration = (timestamp - last_buy_date).days
                
                # Only close the trade if it's profitable
                if profit_dollars > 0:
                    trades.append({
                        'Entry Price': last_buy_price,
                        'Take Profit Price': take_profit_price,
                        'Profit ($)': profit_dollars,
                        'Profit (%)': profit_percent,
                        'Duration (days)': duration,
                        'Entry Date': last_buy_date,
                        'Closing Date': timestamp
                    })
                    signals.append('Take Profit Signal')
                    # Reset for next potential buy
                    last_buy_price = None
                    last_buy_date = None
                else:
                    signals.append('No Action')  # Trade remains open if not profitable
            else:
                signals.append('No Action')  # Trade remains open if no take profit signal

    return signals, trades

# Download stock data
if ticker:
    st.write(f"Fetching data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    if not stock_data.empty:
        close_prices = stock_data["Close"].values

        # Apply Kalman Filter
        kalman_avg = kalman_filter(close_prices)

        # Calculate Z-scores
        z_scores = calculate_z_scores(close_prices, kalman_avg)

        # Generate trading signals and trades
        signals, trades = generate_signals(z_scores, close_prices)

        # Add calculated values to the DataFrame
        stock_data['Kalman Avg'] = kalman_avg
        stock_data['Z-Score'] = np.append([None], z_scores[:-1])  # Adjust Z-score length
        stock_data['Signal'] = [None] + signals

        trades_df = pd.DataFrame(trades)

        # Plotting the results: Price vs Kalman Filter Average
        st.subheader(f'{ticker} Price vs Kalman Filter Average')
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=stock_data.index, y=close_prices, mode='lines', name='Close Prices', line=dict(color='blue')))
        price_fig.add_trace(go.Scatter(x=stock_data.index, y=kalman_avg, mode='lines', name='Kalman Filter Average', line=dict(color='orange')))
        
        # Highlight Buy and Take Profit signals
        buy_signals = stock_data[stock_data['Signal'] == 'Buy Signal']
        take_profit_signals = stock_data[stock_data['Signal'] == 'Take Profit Signal']
        
        price_fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Kalman Avg'], mode='markers', name='Buy Signal', marker=dict(color='green', size=10)))
        price_fig.add_trace(go.Scatter(x=take_profit_signals.index, y=take_profit_signals['Kalman Avg'], mode='markers', name='Take Profit Signal', marker=dict(color='red', size=10)))
        
        price_fig.update_layout(title=f'{ticker} Price vs Kalman Filter Average', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(price_fig)

        # Plotting Z-score
        st.subheader('Z-Score')
        zscore_fig = go.Figure()
        zscore_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Z-Score'], mode='lines', name='Z-Score', line=dict(color='purple')))
        zscore_fig.add_trace(go.Scatter(x=stock_data.index, y=[buy_signal_threshold]*len(stock_data), mode='lines', name=f'Buy Threshold ({buy_signal_threshold})', line=dict(color='red', dash='dash')))
        zscore_fig.add_trace(go.Scatter(x=stock_data.index, y=[take_profit_threshold]*len(stock_data), mode='lines', name=f'Take Profit Threshold ({take_profit_threshold})', line=dict(color='green', dash='dash')))
        
        zscore_fig.update_layout(title='Z-Score Over Time', xaxis_title='Date', yaxis_title='Z-Score')
        st.plotly_chart(zscore_fig)

        # Display trading signals
        st.subheader('Trading Signals')
        st.dataframe(stock_data[['Signal', 'Z-Score', 'Kalman Avg']])
        
        # Display trades
        st.subheader('Trade Performance')
        if not trades_df.empty:
            st.dataframe(trades_df)
        else:
            st.write("No trades executed.")

    else:
        st.error("No data found for this ticker.")
