import pandas as pd
import streamlit as st


# Load Trends data
@st.cache_data
def load_trends_data():
    return pd.read_csv("memecoin_trends.csv", parse_dates=["Date"])

# Load sentiment data
@st.cache_data
def load_sentiment_data():
    return pd.read_csv("memecoin_sentiment_huggingface.csv", parse_dates=["Timestamp"])
