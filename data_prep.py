def prepare_data(): 
  """# 5. Predictive Modeling for Memecoin Price

  This section implements Hidden Markov Model (HMM) for price prediction, integrating sentiment analysis data with historical price data. We'll focus on Dogecoin, Shiba Inu, and Pepe coins.

  ## Approach:
  1. **Data Integration**: Combine price data, sentiment scores, and search trends
  2. **Feature Engineering**: Create meaningful features from multi-modal data
  """

  # Import required libraries
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from datetime import datetime, timedelta
  import warnings
  warnings.filterwarnings('ignore')

  # Machine Learning libraries
  from sklearn.preprocessing import StandardScaler, MinMaxScaler
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  from hmmlearn import hmm

  # Data visualization
  import plotly.graph_objects as go
  import plotly.express as px
  from plotly.subplots import make_subplots

  # Set style for better plots
  plt.style.use('seaborn-v0_8')
  sns.set_palette("husl")

  print("Libraries imported successfully!")

  ## 5.1 Data Loading and Preprocessing

  # Load sentiment data
  sentiment_df = pd.read_csv('memecoin_sentiment_huggingface.csv')
  trends_df = pd.read_csv('memecoin_trends.csv')

  # Convert timestamps to datetime
  sentiment_df['Timestamp'] = pd.to_datetime(sentiment_df['Timestamp'])
  trends_df['Date'] = pd.to_datetime(trends_df['Date'])

  print("Sentiment Data Shape:", sentiment_df.shape)
  print("Trends Data Shape:", trends_df.shape)
  print("\nAvailable coins in sentiment data:")
  print(sentiment_df['Keyword'].value_counts())
  print("\nAvailable coins in trends data:")
  print(trends_df['Keyword'].value_counts())

  ## 5.2 Data Cleaning and Alignment

  # Remove duplicates from sentiment data
  print(f"Before duplicate removal: {len(sentiment_df)} rows")
  sentiment_df = sentiment_df.drop_duplicates(subset=['Cleaned_Text', 'Timestamp', 'Keyword'])
  print(f"After duplicate removal: {len(sentiment_df)} rows")

  # Map keywords to standardized coin names
  coin_mapping = {
      'dogecoin': 'dogecoin',
      'doge': 'dogecoin',
      '$doge': 'dogecoin',
      'shiba': 'shiba_inu',
      'shib': 'shiba_inu',
      '$shiba': 'shiba_inu',
      'shiba coin': 'shiba_inu',
      'shiba inu coin': 'shiba_inu',
      'shibarmy': 'shiba_inu',
      'pepe': 'pepe',
      '$pepe': 'pepe',
      'pepe coin': 'pepe'
  }

  # Standardize coin names in sentiment data
  sentiment_df['coin'] = sentiment_df['Keyword'].map(coin_mapping)
  sentiment_df = sentiment_df.dropna(subset=['coin'])

  # Standardize coin names in trends data
  trends_mapping = {
      'dogecoin': 'dogecoin',
      'shiba inu coin': 'shiba_inu',
      'pepe coin': 'pepe'
  }
  trends_df['coin'] = trends_df['Keyword'].map(trends_mapping)
  trends_df = trends_df.dropna(subset=['coin'])

  print("\nStandardized coin distribution:")
  print("Sentiment data:", sentiment_df['coin'].value_counts())
  print("Trends data:", trends_df['coin'].value_counts())

  ## 5.3 Sentiment Aggregation by Date

  # Extract date from timestamp for aggregation
  sentiment_df['date'] = sentiment_df['Timestamp'].dt.date

  # Aggregate sentiment scores by date and coin
  sentiment_agg = sentiment_df.groupby(['date', 'coin']).agg({
      'Sentiment_Score': ['mean', 'std', 'count'],
      'Label': lambda x: (x == 'POSITIVE').sum() / len(x)  # Positive sentiment ratio
  }).round(4)

  # Flatten column names
  sentiment_agg.columns = ['sentiment_mean', 'sentiment_std', 'sentiment_count', 'positive_ratio']
  sentiment_agg = sentiment_agg.reset_index()

  # Fill missing std with 0 (when count = 1)
  sentiment_agg['sentiment_std'] = sentiment_agg['sentiment_std'].fillna(0)

  print("Sentiment aggregation completed:")
  print(sentiment_agg.head(10))
  print(f"\nDate range: {sentiment_agg['date'].min()} to {sentiment_agg['date'].max()}")

  ## 5.4 Price Data Collection
  import ccxt

  # Function to fetch price data
  def fetch_price_data(symbol, timeframe='1d', limit=600):
      """
      Fetch price data for a given symbol
      """
      try:
          exchange = ccxt.bingx()
          ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
          df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
          df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
          df['date'] = df['timestamp'].dt.date
          return df
      except Exception as e:
          print(f"Error fetching data for {symbol}: {e}")
          return None

  # Fetch price data for memecoins
  coins_symbols = {
      'dogecoin': 'DOGE/USDT',
      'shiba_inu': 'SHIB/USDT',
      'pepe': 'PEPE/USDT'
  }

  price_data = {}
  for coin, symbol in coins_symbols.items():
      print(f"Fetching price data for {coin} ({symbol})...")
      df = fetch_price_data(symbol, timeframe='1d', limit=600)
      if df is not None:
          df['coin'] = coin
          price_data[coin] = df
          print(f"Fetched {len(df)} days of data for {coin}")
      else:
          print(f"Failed to fetch data for {coin}")

  print(f"\nSuccessfully fetched price data for {len(price_data)} coins")

  ## 5.5 Feature Engineering and Data Integration

  # Combine all price data
  all_price_data = []
  for coin, df in price_data.items():
      all_price_data.append(df)

  if all_price_data:
      combined_price = pd.concat(all_price_data, ignore_index=True)
  else:
      # Create synthetic price data for demonstration using actual data date ranges
      print("Creating synthetic price data for demonstration...")
      
      # Get actual date range from sentiment and trends data
      sentiment_min_date = sentiment_df['Timestamp'].min().date()
      sentiment_max_date = sentiment_df['Timestamp'].max().date()
      trends_min_date = trends_df['Date'].min().date()
      trends_max_date = trends_df['Date'].max().date()
      
      # Use the overlapping date range
      start_date = max(sentiment_min_date, trends_min_date)
      end_date = min(sentiment_max_date, trends_max_date)
      
      print(f"Using date range from {start_date} to {end_date}")
      
      # Generate dates for the actual data range
      dates = pd.date_range(start=start_date, end=end_date, freq='D')

      synthetic_data = []
      for coin in ['dogecoin', 'shiba_inu', 'pepe']:
          # Generate realistic price movements
          np.random.seed(42 if coin == 'dogecoin' else 43 if coin == 'shiba_inu' else 44)
          price_base = 0.25 if coin == 'dogecoin' else 0.00002 if coin == 'shiba_inu' else 0.000008
          prices = []
          current_price = price_base

          for i in range(len(dates)):
              # Add some trend and volatility
              change = np.random.normal(0.001, 0.05)  # Daily change
              current_price *= (1 + change)
              prices.append(current_price)

          for i, date in enumerate(dates):
              price = prices[i]
              synthetic_data.append({
                  'timestamp': pd.Timestamp(date),
                  'date': date.date(),
                  'open': price * 0.99,
                  'high': price * 1.02,
                  'low': price * 0.98,
                  'close': price,
                  'volume': np.random.uniform(1000000, 10000000),
                  'coin': coin
              })

      combined_price = pd.DataFrame(synthetic_data)

  print(f"Combined price data shape: {combined_price.shape}")
  print(f"Price data date range: {combined_price['date'].min()} to {combined_price['date'].max()}")

  ## 5.6 Technical Indicators and Price Features

  def calculate_technical_indicators(df):
      """
      Calculate technical indicators for price data
      """
      df = df.copy().sort_values('timestamp')

      # Price-based features
      df['price_change'] = df['close'].pct_change()
      df['price_volatility'] = df['price_change'].rolling(window=7).std()
      df['price_momentum'] = df['close'].rolling(window=5).mean() / df['close'].rolling(window=20).mean()

      # Simple moving averages
      df['sma_5'] = df['close'].rolling(window=5).mean()
      df['sma_20'] = df['close'].rolling(window=20).mean()
      df['sma_signal'] = (df['sma_5'] > df['sma_20']).astype(int)

      # Volume features
      df['volume_change'] = df['volume'].pct_change()
      df['volume_ma'] = df['volume'].rolling(window=7).mean()

      # Price range features
      df['high_low_ratio'] = df['high'] / df['low']
      df['close_range_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

      return df

  # Calculate technical indicators for each coin
  enhanced_price_data = []
  for coin in combined_price['coin'].unique():
      coin_data = combined_price[combined_price['coin'] == coin].copy()
      coin_data = calculate_technical_indicators(coin_data)
      enhanced_price_data.append(coin_data)

  enhanced_price = pd.concat(enhanced_price_data, ignore_index=True)

  print("Technical indicators calculated successfully!")
  print(f"Enhanced price data shape: {enhanced_price.shape}")
  print("New columns:", [col for col in enhanced_price.columns if col not in combined_price.columns])

  # Ensure date columns are datetime for proper merging
  sentiment_agg['date'] = pd.to_datetime(sentiment_agg['date'])
  trends_df['date'] = pd.to_datetime(trends_df['Date']).dt.date
  trends_df['date'] = pd.to_datetime(trends_df['date'])  # Convert back to datetime
  enhanced_price['date'] = pd.to_datetime(enhanced_price['date'])

  # Normalize sentiment_mean to 0–1 range from -1 to 1
  sentiment_agg['sentiment_mean'] = (sentiment_agg['sentiment_mean'] + 1) / 2  # Now in range [0,1]

  # Normalize sentiment_std if needed (optional, e.g., dividing by max std to scale)
  max_sentiment_std = sentiment_agg['sentiment_std'].max()
  if max_sentiment_std > 0:
      sentiment_agg['sentiment_std'] = sentiment_agg['sentiment_std'] / max_sentiment_std
  else:
      sentiment_agg['sentiment_std'] = 0

  # INTERPOLATE missing sentiment values per coin over dates
  sentiment_agg = sentiment_agg.sort_values(['coin', 'date'])

  # Create a complete date range per coin based on actual price data
  full_dates = (
      enhanced_price[['coin', 'date']]
      .drop_duplicates()
      .sort_values(['coin', 'date'])
  )

  # Merge full date range with sentiment
  sentiment_full = pd.merge(full_dates, sentiment_agg, on=['coin', 'date'], how='left')

  # Interpolate missing sentiment values per coin
  sentiment_columns = ['sentiment_mean', 'sentiment_std', 'sentiment_count', 'positive_ratio']
  for col in sentiment_columns:
      sentiment_full[col] = (
          sentiment_full
          .groupby('coin')[col]
          .transform(lambda group: group.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill'))
      )

  # Ensure trends_df has the right date format and merge
  final_data = enhanced_price.merge(
      sentiment_full, on=['date', 'coin'], how='left'
  ).merge(
      trends_df[['date', 'coin', 'Search_Score']], on=['date', 'coin'], how='left'
  )

  # Fill missing Search_Score with per-coin median, then forward fill
  for coin in final_data['coin'].unique():
      coin_mask = final_data['coin'] == coin
      median_score = final_data.loc[coin_mask, 'Search_Score'].median()
      if pd.isna(median_score):
          median_score = 50  # Default search score
      final_data.loc[coin_mask, 'Search_Score'] = (
          final_data.loc[coin_mask, 'Search_Score']
          .fillna(median_score)
          .interpolate(method='linear')
          .fillna(method='bfill')
          .fillna(method='ffill')
      )

  # Target variable: next day price change
  final_data = final_data.sort_values(['coin', 'date'])
  final_data['target'] = final_data.groupby('coin')['close'].shift(-1)
  final_data['target_change'] = (final_data['target'] - final_data['close']) / final_data['close']

  # Drop rows without target
  final_data = final_data.dropna(subset=['target'])

  # Print dataset info
  print(f"Final integrated dataset shape: {final_data.shape}")
  print(f"Date range: {final_data['date'].min()} to {final_data['date'].max()}")
  print(f"Coins available: {final_data['coin'].value_counts()}")
  print("\nSample of final dataset:")
  print(final_data[['date', 'coin', 'close', 'sentiment_mean', 'positive_ratio', 'Search_Score', 'target_change']].head(10))

  final_data.describe()
  
  return final_data