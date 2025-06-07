import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

from helper import get_crypto_data_ccxt

class StockPricePredictor:
    def __init__(self, model_type='ensemble', lookback_period=60):
        self.model_type = model_type
        self.lookback_period = lookback_period
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.is_trained = False
        
    def create_features(self, df):
        """Create technical indicators and features for prediction"""
        df = df.copy()
        
        # Technical indicators
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['Bollinger_upper'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
        df['Bollinger_lower'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def check_date_overlap(self, price_df, other_df, other_name, date_col='Date'):
        """Check for date overlap and provide debugging information"""
        price_dates = set(pd.to_datetime(price_df['Date']).dt.date)
        other_dates = set(pd.to_datetime(other_df[date_col]).dt.date)
        
        price_min = min(price_dates)
        price_max = max(price_dates)
        other_min = min(other_dates)
        other_max = max(other_dates)
        
        overlap = price_dates & other_dates
        
        print(f"\n=== DATE OVERLAP ANALYSIS FOR {other_name.upper()} ===")
        print(f"Price data: {len(price_dates)} dates from {price_min} to {price_max}")
        print(f"{other_name} data: {len(other_dates)} dates from {other_min} to {other_max}")
        print(f"Overlapping dates: {len(overlap)}")
        
        if len(overlap) == 0:
            print(f"WARNING: No overlapping dates!")
            print(f"Price data ends {(other_min - price_max).days} days before {other_name} starts")
            print(f"Or {other_name} ends {(price_min - other_max).days} days before price starts")
            
            # Show sample dates from each dataset
            print(f"\nSample price dates: {sorted(list(price_dates))[:5]} ... {sorted(list(price_dates))[-5:]}")
            print(f"Sample {other_name} dates: {sorted(list(other_dates))[:5]} ... {sorted(list(other_dates))[-5:]}")
        else:
            print(f"Found {len(overlap)} overlapping dates")
            overlap_sorted = sorted(list(overlap))
            print(f"Overlap range: {overlap_sorted[0]} to {overlap_sorted[-1]}")
        
        return len(overlap) > 0

    def add_sentiment_features(self, price_df, sentiment_df):
        """Add sentiment features to price data"""
        print(f"Price DataFrame shape: {price_df.shape}")
        print(f"Price date range: {price_df['Date'].min()} to {price_df['Date'].max()}")
        
        if sentiment_df is not None and not sentiment_df.empty:
            print(f"Sentiment DataFrame shape: {sentiment_df.shape}")
            print(f"Sentiment columns: {sentiment_df.columns.tolist()}")
            
            # Ensure Timestamp column exists and is datetime
            if 'Timestamp' not in sentiment_df.columns:
                print("Warning: 'Timestamp' column not found in sentiment data")
                # Add default sentiment columns
                price_df['Sentiment_Mean'] = 0.0
                price_df['Sentiment_Std'] = 0.0
                price_df['Sentiment_Count'] = 0.0
                price_df['Positive_Ratio'] = 0.5
                return price_df
            
            # Convert Timestamp to datetime if not already
            sentiment_df = sentiment_df.copy()
            sentiment_df['Timestamp'] = pd.to_datetime(sentiment_df['Timestamp'])
            
            # Ensure Date column in price_df is datetime
            price_df = price_df.copy()
            price_df['Date'] = pd.to_datetime(price_df['Date'])
            
            # Extract date from timestamp for grouping
            sentiment_df['Date_only'] = sentiment_df['Timestamp'].dt.date
            
            print(f"Sentiment date range: {sentiment_df['Date_only'].min()} to {sentiment_df['Date_only'].max()}")
            
            # Check for date overlap before processing
            has_overlap = self.check_date_overlap(price_df, sentiment_df, "sentiment", "Timestamp")
            
            # Check if required columns exist
            required_cols = ['Sentiment_Score', 'Label']
            missing_cols = [col for col in required_cols if col not in sentiment_df.columns]
            if missing_cols:
                print(f"Warning: Missing columns in sentiment data: {missing_cols}")
                # Add default values for missing columns
                if 'Sentiment_Score' not in sentiment_df.columns:
                    sentiment_df['Sentiment_Score'] = 0.0
                if 'Label' not in sentiment_df.columns:
                    sentiment_df['Label'] = 'NEUTRAL'
            
            # Only try to merge if there's potential overlap or if we want to try anyway
            if has_overlap or len(sentiment_df) > 0:
                try:
                    # Aggregate sentiment by date
                    sentiment_daily = sentiment_df.groupby('Date_only').agg({
                        'Sentiment_Score': ['mean', 'std', 'count'],
                        'Label': lambda x: (x == 'POSITIVE').sum() / len(x) if len(x) > 0 else 0.5
                    }).reset_index()
                    
                    # Flatten column names
                    sentiment_daily.columns = ['Date', 'Sentiment_Mean', 'Sentiment_Std', 'Sentiment_Count', 'Positive_Ratio']
                    
                    # Convert Date back to datetime for merging
                    sentiment_daily['Date'] = pd.to_datetime(sentiment_daily['Date'])
                    
                    print(f"Aggregated sentiment data shape: {sentiment_daily.shape}")
                    print(f"Sample aggregated data:\n{sentiment_daily.head()}")
                    
                    # Check for NaN values in aggregated data
                    nan_counts = sentiment_daily.isnull().sum()
                    if nan_counts.any():
                        print(f"NaN values in aggregated sentiment data:\n{nan_counts}")
                        # Fill NaN values in std column (happens when only one value per day)
                        sentiment_daily['Sentiment_Std'] = sentiment_daily['Sentiment_Std'].fillna(0)
                    
                    # Merge with price data using left join
                    original_shape = price_df.shape
                    price_df = price_df.merge(sentiment_daily, on='Date', how='left')
                    print(f"Price DataFrame shape after merge: {price_df.shape} (was {original_shape})")
                    
                    # Fill missing sentiment data with default values
                    sentiment_cols = ['Sentiment_Mean', 'Sentiment_Std', 'Sentiment_Count', 'Positive_Ratio']
                    for col in sentiment_cols:
                        if col in price_df.columns:
                            before_fill = price_df[col].isnull().sum()
                            if col == 'Sentiment_Mean':
                                # Use actual mean from available data if exists, else 0
                                fill_value = sentiment_daily['Sentiment_Mean'].mean() if not sentiment_daily['Sentiment_Mean'].isnull().all() else 0.0
                                price_df[col] = price_df[col].fillna(fill_value)
                            elif col == 'Sentiment_Std':
                                price_df[col] = price_df[col].fillna(0)
                            elif col == 'Sentiment_Count':
                                price_df[col] = price_df[col].fillna(0)
                            elif col == 'Positive_Ratio':
                                # Use actual mean from available data if exists, else 0.5
                                fill_value = sentiment_daily['Positive_Ratio'].mean() if not sentiment_daily['Positive_Ratio'].isnull().all() else 0.5
                                price_df[col] = price_df[col].fillna(fill_value)
                        
                        after_fill = price_df[col].isnull().sum()
                        if before_fill > 0:
                            print(f"Filled {before_fill} NaN values in {col} with value {fill_value if 'fill_value' in locals() else 'default'}")
                    
                    # Final check for any remaining NaN values
                    final_nan_counts = price_df[sentiment_cols].isnull().sum()
                    if final_nan_counts.any():
                        print(f"Warning: Remaining NaN values after filling:\n{final_nan_counts}")
                    
                except Exception as e:
                    print(f"Error during sentiment aggregation: {e}")
                    print("Adding default sentiment columns...")
                    # Add default sentiment columns if aggregation fails
                    price_df['Sentiment_Mean'] = 0.0
                    price_df['Sentiment_Std'] = 0.0
                    price_df['Sentiment_Count'] = 0.0
                    price_df['Positive_Ratio'] = 0.5
            else:
                print("No date overlap found, adding default sentiment columns...")
                price_df['Sentiment_Mean'] = 0.0
                price_df['Sentiment_Std'] = 0.0
                price_df['Sentiment_Count'] = 0.0
                price_df['Positive_Ratio'] = 0.5
        else:
            print("No sentiment data provided, adding default sentiment columns...")
            # Add default sentiment columns when no data is provided
            price_df['Sentiment_Mean'] = 0.0
            price_df['Sentiment_Std'] = 0.0
            price_df['Sentiment_Count'] = 0.0
            price_df['Positive_Ratio'] = 0.5
        
        print(f"Final price DataFrame shape: {price_df.shape}")
        return price_df
    
    def add_trends_features(self, price_df, trends_df):
        """Add Google Trends features to price data"""
        print(f"Adding trends features - Price DataFrame shape: {price_df.shape}")
        
        if trends_df is not None and not trends_df.empty:
            print(f"Trends DataFrame shape: {trends_df.shape}")
            print(f"Trends columns: {trends_df.columns.tolist()}")
            
            # Ensure required columns exist
            if 'Date' not in trends_df.columns:
                print("Warning: 'Date' column not found in trends data")
                # Add default trends columns
                price_df['Trends_Mean'] = 50.0  # Default search score
                price_df['Trends_Max'] = 50.0
                price_df['Trends_Std'] = 0.0
                return price_df
            
            if 'Search_Score' not in trends_df.columns:
                print("Warning: 'Search_Score' column not found in trends data")
                # Add default trends columns
                price_df['Trends_Mean'] = 50.0
                price_df['Trends_Max'] = 50.0
                price_df['Trends_Std'] = 0.0
                return price_df
            
            # Ensure both DataFrames have consistent datetime format
            trends_df = trends_df.copy()
            price_df = price_df.copy()
            
            trends_df['Date'] = pd.to_datetime(trends_df['Date'])
            price_df['Date'] = pd.to_datetime(price_df['Date'])
            
            print(f"Trends date range: {trends_df['Date'].min()} to {trends_df['Date'].max()}")
            print(f"Price date range: {price_df['Date'].min()} to {price_df['Date'].max()}")
            
            # Check for date overlap
            has_overlap = self.check_date_overlap(price_df, trends_df, "trends", "Date")
            
            if has_overlap or len(trends_df) > 0:
                try:
                    # Aggregate trends by date
                    trends_daily = trends_df.groupby('Date').agg({
                        'Search_Score': ['mean', 'max', 'std']
                    }).reset_index()
                    
                    # Flatten column names
                    trends_daily.columns = ['Date', 'Trends_Mean', 'Trends_Max', 'Trends_Std']
                    
                    print(f"Aggregated trends data shape: {trends_daily.shape}")
                    print(f"Sample aggregated trends data:\n{trends_daily.head()}")
                    
                    # Handle NaN values in aggregated data
                    trends_daily['Trends_Std'] = trends_daily['Trends_Std'].fillna(0)
                    
                    # Merge with price data
                    original_shape = price_df.shape
                    price_df = price_df.merge(trends_daily, on='Date', how='left')
                    print(f"Price DataFrame shape after trends merge: {price_df.shape} (was {original_shape})")
                    
                    # Fill missing trends data with appropriate defaults
                    trends_cols = ['Trends_Mean', 'Trends_Max', 'Trends_Std']
                    for col in trends_cols:
                        if col in price_df.columns:
                            before_fill = price_df[col].isnull().sum()
                            if col in ['Trends_Mean', 'Trends_Max']:
                                # Use global mean or default to 50 if no data
                                fill_value = trends_daily[col].mean() if not trends_daily[col].isnull().all() else 50.0
                                price_df[col] = price_df[col].fillna(fill_value)
                            elif col == 'Trends_Std':
                                fill_value = 0.0
                                price_df[col] = price_df[col].fillna(fill_value)
                        
                        after_fill = price_df[col].isnull().sum()
                        if before_fill > 0:
                            print(f"Filled {before_fill} NaN values in {col} with value {fill_value}")
                
                except Exception as e:
                    print(f"Error during trends aggregation: {e}")
                    print("Adding default trends columns...")
                    # Add default trends columns if aggregation fails
                    price_df['Trends_Mean'] = 50.0
                    price_df['Trends_Max'] = 50.0
                    price_df['Trends_Std'] = 0.0
            else:
                print("No date overlap found, adding default trends columns...")
                price_df['Trends_Mean'] = 50.0
                price_df['Trends_Max'] = 50.0
                price_df['Trends_Std'] = 0.0
        else:
            print("No trends data provided, adding default trends columns...")
            # Add default trends columns when no data is provided
            price_df['Trends_Mean'] = 50.0
            price_df['Trends_Max'] = 50.0
            price_df['Trends_Std'] = 0.0
        
        print(f"Final price DataFrame shape after trends: {price_df.shape}")
        return price_df
    
    def prepare_data_for_training(self, df):
        """Prepare data for model training"""
        # Select feature columns (exclude Date and target)
        feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]
        self.feature_columns = feature_cols
        
        X = df[feature_cols].values
        y = df['Close'].values
        
        return X, y
    
    def create_lstm_sequences(self, X, y):
        """Create sequences for LSTM model"""
        X_seq, y_seq = [], []
        
        for i in range(self.lookback_period, len(X)):
            X_seq.append(X[i-self.lookback_period:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train(self, df, test_size=0.2):
        """Train the prediction models"""
        # Prepare data
        print("Ahmed is", df)
        df = self.create_features(df)
        # print("DF is: ", df.describe())
        X, y = self.prepare_data_for_training(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
        
        # Scale data
        self.scalers['X'] = StandardScaler()
        self.scalers['y'] = MinMaxScaler()
        
        X_train_scaled = self.scalers['X'].fit_transform(X_train)
        X_test_scaled = self.scalers['X'].transform(X_test)
        y_train_scaled = self.scalers['y'].fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.scalers['y'].transform(y_test.reshape(-1, 1)).flatten()
        
        # Initialize models
        models_to_train = []
        if self.model_type == 'ensemble':
            models_to_train = ['lstm', 'svr', 'random_forest']
        else:
            models_to_train = [self.model_type]
        
        evaluation_results = {}
        
        for model_name in models_to_train:
            print(f"Training {model_name.upper()} model...")
            
            if model_name == 'lstm':
                # LSTM model
                X_train_seq, y_train_seq = self.create_lstm_sequences(X_train_scaled, y_train_scaled)
                X_test_seq, y_test_seq = self.create_lstm_sequences(X_test_scaled, y_test_scaled)
                
                if len(X_train_seq) > 0:
                    model = self.build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
                    
                    # Early stopping
                    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=10, restore_best_weights=True)
                    
                    history = model.fit(
                        X_train_seq, y_train_seq,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    
                    self.models['lstm'] = model
                    
                    
                    # Evaluate
                    if len(X_test_seq) > 0:
                        y_pred_scaled = model.predict(X_test_seq, verbose=0)
                        y_pred = self.scalers['y'].inverse_transform(y_pred_scaled).flatten()
                        y_true = self.scalers['y'].inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
                        
                        evaluation_results['lstm'] = self.calculate_metrics(y_true, y_pred)
            
            elif model_name == 'svr':
                # SVR model
                model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
                model.fit(X_train_scaled, y_train_scaled)
                self.models['svr'] = model
                
                # Evaluate
                y_pred_scaled = model.predict(X_test_scaled)
                y_pred = self.scalers['y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                y_true = self.scalers['y'].inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
                
                evaluation_results['svr'] = self.calculate_metrics(y_true, y_pred)
            
            elif model_name == 'random_forest':
                # Random Forest model
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train_scaled)
                self.models['random_forest'] = model
                
                # Evaluate
                y_pred_scaled = model.predict(X_test_scaled)
                y_pred = self.scalers['y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                y_true = self.scalers['y'].inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
                
                evaluation_results['random_forest'] = self.calculate_metrics(y_true, y_pred)
        
        self.is_trained = True
        
        # Return average metrics if ensemble
        if len(evaluation_results) > 1:
            avg_metrics = {}
            for metric in ['rmse', 'mae', 'r2']:
                avg_metrics[metric] = np.mean([result[metric] for result in evaluation_results.values()])
            return avg_metrics
        else:
            return list(evaluation_results.values())[0] if evaluation_results else {}
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def predict(self, df, days_ahead=7):
        """Make predictions using trained models"""
        if not self.is_trained or not self.models:
            raise ValueError("Models not trained. Please train models first.")
        
        # Prepare data
        df = self.create_features(df)
        X = df[self.feature_columns].values
        
        if len(X) == 0:
            raise ValueError("No data available for prediction")
        
        # Scale data
        X_scaled = self.scalers['X'].transform(X)
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'lstm':
                    # LSTM prediction
                    if len(X_scaled) >= self.lookback_period:
                        X_seq = X_scaled[-self.lookback_period:].reshape(1, self.lookback_period, -1)
                        pred_scaled = model.predict(X_seq, verbose=0)
                        pred = self.scalers['y'].inverse_transform(pred_scaled).flatten()
                        
                        # Generate multi-step predictions
                        multi_pred = []
                        current_seq = X_seq.copy()
                        
                        for _ in range(days_ahead):
                            next_pred_scaled = model.predict(current_seq, verbose=0)
                            next_pred = self.scalers['y'].inverse_transform(next_pred_scaled).flatten()[0]
                            multi_pred.append(next_pred)
                            
                            # Update sequence (simplified approach)
                            new_features = current_seq[0, -1, :].copy()
                            new_features[0] = next_pred_scaled[0, 0]  # Update price feature
                            current_seq = np.roll(current_seq, -1, axis=1)
                            current_seq[0, -1, :] = new_features
                        
                        predictions[model_name] = multi_pred
                
                else:
                    # SVR and Random Forest predictions
                    last_features = X_scaled[-1:] 
                    pred_scaled = model.predict(last_features)
                    pred = self.scalers['y'].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                    
                    # For multi-step, use the same prediction (simplified approach)
                    predictions[model_name] = [pred[0]] * days_ahead
                    
            except Exception as e:
                print(f"Error in {model_name} prediction: {e}")
                continue
        
        # If ensemble, return individual predictions
        if len(predictions) > 1:
            return predictions
        else:
            return list(predictions.values())[0] if predictions else [0] * days_ahead
    
    def save_model(self, filepath):
        """Save trained models"""
        model_data = {
            'model_type': self.model_type,
            'lookback_period': self.lookback_period,
            'feature_columns': self.feature_columns,
            'scalers': self.scalers,
            'is_trained': self.is_trained
        }
        
        # Save non-LSTM models
        for name, model in self.models.items():
            if name != 'lstm':
                model_data[f'{name}_model'] = model
        
        # Save to pickle
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save LSTM model separately if it exists
        if 'lstm' in self.models:
            self.models['lstm'].save(f"{filepath}_lstm.h5")
    
    def load_models(self, filepath):
        """Load trained models"""
        try:
            # Load pickle data
            with open(f"{filepath}.pkl", 'rb') as f:
                model_data = pickle.load(f)
            
            self.model_type = model_data['model_type']
            self.lookback_period = model_data['lookback_period']
            self.feature_columns = model_data['feature_columns']
            self.scalers = model_data['scalers']
            self.is_trained = model_data['is_trained']
            
            # Load non-LSTM models
            self.models = {}
            for key, value in model_data.items():
                if key.endswith('_model'):
                    model_name = key.replace('_model', '')
                    self.models[model_name] = value
            
            # Load LSTM model if it exists
            try:
                lstm_model = tf.keras.models.load_model(f"{filepath}_lstm.h5")
                self.models['lstm'] = lstm_model
            except:
                print("LSTM model file not found, skipping...")
            
            print(f"Loaded models: {list(self.models.keys())}")
            
        except Exception as e:
            raise Exception(f"Failed to load models: {e}")

def create_sample_price_data(sentiment_df, trends_df):
    """Create sample price data or fetch from ccxt"""
    try:
        # Try to fetch real data from ccxt
        price_df = get_crypto_data_ccxt('DOGE/USDT', 'binance', '1d', 365)
        print("Fetched real price data from Binance")
        return price_df
    except Exception as e:
        print(f"Failed to fetch real data: {e}")
        print("Creating sample price data...")
        
        # Create sample data based on date range from sentiment/trends data
        start_date = pd.to_datetime('2024-01-01')
        end_date = pd.to_datetime('2024-12-31')
        
        if sentiment_df is not None and not sentiment_df.empty:
            start_date = sentiment_df['Timestamp'].min()
            end_date = sentiment_df['Timestamp'].max()
        elif trends_df is not None and not trends_df.empty:
            start_date = trends_df['Date'].min()
            end_date = trends_df['Date'].max()
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic price data
        initial_price = 0.08
        prices = [initial_price]
        
        for i in range(len(dates) - 1):
            change = np.random.normal(0, 0.03)  # 3% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.001))
        
        data = []
        for i, date in enumerate(dates):
            price = prices[i]
            data.append({
                'Date': date,
                'Open': price * np.random.uniform(0.99, 1.01),
                'High': price * np.random.uniform(1.02, 1.08),
                'Low': price * np.random.uniform(0.92, 0.98),
                'Close': price,
                'Volume': np.random.randint(5000000, 50000000)
            })
        
        return pd.DataFrame(data)

def combine_all_data(price_df, sentiment_df, trends_df):
    """Combine price, sentiment, and trends data"""
    # Start with price data
    combined_df = price_df.copy()
    
    # Initialize predictor to add features
    predictor = StockPricePredictor()
    
    # Add sentiment features
    if sentiment_df is not None and not sentiment_df.empty:
        combined_df = predictor.add_sentiment_features(combined_df, sentiment_df)
    
    # Add trends features
    if trends_df is not None and not trends_df.empty:
        combined_df = predictor.add_trends_features(combined_df, trends_df)
    
    # Sort by date
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    return combined_df
