import os
from datetime import datetime, timedelta

import ccxt
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data
def load_sentiment_data():
    """Load sentiment data from CSV file"""
    file_path = "memecoin_sentiment_huggingface.csv"
    try:
        df = pd.read_csv(file_path)
        # Ensure proper column names and types
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        elif 'Date' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Date'])
        
        # Standardize column names
        required_columns = ['Timestamp', 'Source', 'Keyword', 'Original_Text', 'Cleaned_Text', 'Sentiment_Score', 'Label']
        for col in required_columns:
            if col not in df.columns:
                if col == 'Original_Text' and 'Text' in df.columns:
                    df['Original_Text'] = df['Text']
                elif col == 'Cleaned_Text' and 'Text' in df.columns:
                    df['Cleaned_Text'] = df['Text']
                elif col == 'Sentiment_Score':
                    df['Sentiment_Score'] = np.random.uniform(-1, 1, len(df))
                elif col == 'Label':
                    df['Label'] = np.random.choice(['POSITIVE', 'NEGATIVE', 'NEUTRAL'], len(df))
        
        return df
    except Exception as e:
        # Create sample data if file doesn't exist
        print(f"Warning: Could not load sentiment data: {e}")
        return create_sample_sentiment_data()

@st.cache_data
def load_trends_data():
    """Load Google Trends data from CSV file"""
    file_path = "memecoin_trends.csv"
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        return df
    except Exception as e:
        print(f"Warning: Could not load trends data: {e}")
        return create_sample_trends_data()

def get_crypto_data_ccxt(symbol='DOGE/USDT', exchange_name='binance', timeframe='1d', limit=100):
    """
    Fetch cryptocurrency data using ccxt with proper date handling
    """
    try:
        exchange = ccxt.bingx()
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        print(f"Fetched {len(df)} records for {symbol}")
        
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop('timestamp', axis=1)
        
        # Rename columns to match expected format
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Ensure Date column is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        return df
        
    except Exception as e:
        print(f"Error fetching data from {exchange_name}: {e}")
        print("Using fallback data...")
        return create_sample_price_data_fallback()

def create_sample_sentiment_data():
    """Create sample sentiment data"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    keywords = ['dogecoin', 'shiba', 'pepe', 'floki']
    sources = ['Twitter', 'Reddit', 'Telegram']
    
    data = []
    for date in dates[:100]:  # Limit to 100 entries
        for keyword in keywords:
            sentiment_score = np.random.uniform(-1, 1)
            label = 'POSITIVE' if sentiment_score > 0.2 else 'NEGATIVE' if sentiment_score < -0.2 else 'NEUTRAL'
            
            data.append({
                'Timestamp': date,
                'Source': np.random.choice(sources),
                'Keyword': keyword,
                'Original_Text': f"Sample text about {keyword}",
                'Cleaned_Text': f"sample text {keyword}",
                'Sentiment_Score': sentiment_score,
                'Label': label
            })
    
    return pd.DataFrame(data)

def create_sample_trends_data():
    """Create sample Google Trends data"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
    keywords = ['dogecoin', 'shiba', 'pepe', 'floki']
    
    data = []
    for date in dates:
        for keyword in keywords:
            data.append({
                'Date': date,
                'Source': 'Google Trends',
                'Keyword': keyword,
                'Search_Score': np.random.randint(1, 100)
            })
    
    return pd.DataFrame(data)

def create_sample_price_data_fallback():
    """Create fallback sample price data with date alignment"""
    # Use consistent date range
    start_date = '2024-05-19'  # Match the trends data start
    end_date = '2025-05-25'    # Match the trends data end
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic price data with random walk
    np.random.seed(42)  # Fixed seed for reproducibility
    initial_price = 0.1
    prices = [initial_price]
    
    for i in range(len(dates) - 1):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.001))  # Ensure price doesn't go negative
    
    data = []
    for i, date in enumerate(dates):
        price = prices[i]
        data.append({
            'Date': date,
            'Open': price * np.random.uniform(0.98, 1.02),
            'High': price * np.random.uniform(1.01, 1.05),
            'Low': price * np.random.uniform(0.95, 0.99),
            'Close': price,
            'Volume': np.random.randint(1000000, 10000000)
        })
    
    return pd.DataFrame(data)
