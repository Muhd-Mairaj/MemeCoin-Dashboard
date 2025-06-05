import ccxt
import numpy as np
import pandas as pd
import streamlit as st
import ta
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Memecoin Mania", layout="wide")

# ---------------------- Sidebar Config ----------------------
st.sidebar.title("⚙️ Configuration")

symbol = st.sidebar.selectbox("Choose Memecoin", ["DOGE/USDT", "SHIB/USDT", "PEPE/USDT"])
timeframe = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"])
limit = st.sidebar.slider("Data Points", 100, 1000, 500)

look_back = st.sidebar.slider("LSTM Look-back Window", 10, 100, 60)
future_steps = st.sidebar.slider("LSTM Future Steps", 1, 50, 10)
epochs = st.sidebar.slider("LSTM Epochs", 5, 100, 20)


# ---------------------- Data Fetching ----------------------
@st.cache_data(show_spinner=False)
def fetch_data(symbol, timeframe, limit):
    exchange = ccxt.bingx()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

df = fetch_data(symbol, timeframe, limit)


# ---------------------- Tabs ----------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Technical Indicators", "🔮 LSTM Prediction", "🧠 Insights"])


# ---------------------- Tab 1: Price Chart ----------------------
with tab1:
    st.header("📈 Historical Close Price")
    st.line_chart(df['close'])
    st.write(df.tail())


# ---------------------- Tab 2: Technical Indicators ----------------------
with tab2:
    st.header("📊 Technical Indicators")

    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['RSI'] = ta.momentum.RSIIndicator(close=df['close']).rsi()

    ind_tab1, ind_tab2 = st.tabs(["SMA (20)", "RSI"])

    with ind_tab1:
        st.subheader("Simple Moving Average (20)")
        st.line_chart(df[['close', 'SMA_20']].dropna())

    with ind_tab2:
        st.subheader("Relative Strength Index")
        st.line_chart(df[['RSI']].dropna())


# ---------------------- Tab 3: LSTM Prediction ----------------------
with tab3:
    st.header("🔮 LSTM Price Prediction")

    close_prices = df[['close']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(len(scaled) - look_back - future_steps):
        X.append(scaled[i:i + look_back])
        y.append(scaled[i + look_back:i + look_back + future_steps, 0])
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        st.warning("Not enough data for training. Increase limit or reduce look-back.")
    else:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            LSTM(50),
            Dense(future_steps)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

        last_sequence = scaled[-look_back:]
        prediction = model.predict(np.expand_dims(last_sequence, axis=0))[0]
        prediction_rescaled = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()

        future_dates = pd.date_range(start=df.index[-1], periods=future_steps + 1, freq='1H')[1:]
        prediction_df = pd.DataFrame({'Predicted Close': prediction_rescaled}, index=future_dates)

        st.subheader("📊 Predicted Future Prices")
        st.line_chart(prediction_df)


# ---------------------- Tab 4: Insights ----------------------
with tab4:
    st.header("🧠 Key Insights")
    st.markdown("""
- **Price Patterns**: Visualize how memecoins behave over time.
- **Indicators**: RSI reveals overbought/oversold zones; SMA shows smoothed trends.
- **LSTM**: Works surprisingly well despite memecoins' chaotic nature—short-term signals are still learnable!
- **Dashboard Use**: Use this for interactive presentations, rapid exploration, and live updates from crypto exchanges.

---  
*Created with ❤️ by G5*
""")
