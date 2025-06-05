import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
# Load helper functions
from helper import load_trends_data, load_sentiment_data

st.set_page_config(page_title="Memecoin Sentiment & Price Analytics", layout="wide")

# Sidebar navigation
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "1. Sentiment Analysis",
    "2. Search Trends",
    "3. Price Correlation",
    "4. Price Prediction",
    "5. Memecoins vs Traditional Coins",
    "6. Raw Data Explorer"
])

# Render different pages
if page == "Overview":
    st.title("🚀 Memecoin Sentiment & Price Prediction Dashboard")
    st.markdown("""
    Welcome to the **Memecoin Analytics Dashboard**!
    This dashboard presents a comprehensive analysis pipeline for memecoins:

    1. **Sentiment Analysis** of social media posts
    2. **Google Trends** search interest
    3. **Correlation** of sentiment and search trends with price
    4. **Forecasting** future prices using LSTM & HMM models
    5. **Comparing** memecoins and traditional cryptocurrencies
    6. **Exploring Raw Data** in one place
    """)

elif page == "1. Sentiment Analysis":
    # Load sentiment data
    sentiment_df = load_sentiment_data()

    st.title("📊 Sentiment Analysis of Social Media Posts")
    st.markdown("Sentiment classification (Positive / Neutral / Negative) of posts related to specific memecoins.")

    # Sidebar filters
    st.subheader("🔍 Filters")
    keywords = sorted(sentiment_df["Keyword"].unique().tolist())
    selected_keywords = st.multiselect("Select Keyword(s)", keywords, default=keywords[:0])

    min_date, max_date = sentiment_df["Timestamp"].min(), sentiment_df["Timestamp"].max()
    start_date, end_date = st.date_input("Select Time Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    # Filter data
    if not selected_keywords:
        selected_keywords = keywords

    # Sentiment type filter
    sentiment_labels = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    selected_labels = st.multiselect("Filter by Sentiment Type", sentiment_labels, default=sentiment_labels)

    # Filter the DataFrame based on user selections
    filtered_df = sentiment_df[
        (sentiment_df["Keyword"].isin(selected_keywords)) &
        (sentiment_df["Timestamp"] >= pd.to_datetime(start_date)) &
        (sentiment_df["Timestamp"] <= pd.to_datetime(end_date))
    ]

    # Apply sentiment filter
    filtered_df = filtered_df[filtered_df["Label"].isin(selected_labels)]

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        st.stop()

    # Sentiment Distribution Pie Chart
    title_kw = ", ".join(selected_keywords) if selected_keywords else "Selected Keywords"
    sentiment_colors = {
        "POSITIVE": "#66c2a5",
        "NEGATIVE": "#fc8d62",
        "NEUTRAL": "#8da0cb"
    }
    pie_fig = px.pie(
        filtered_df,
        names="Label",
        title=f"Sentiment Distribution for {title_kw}",
        hole=0.4,
        color="Label",
        color_discrete_map=sentiment_colors
    )
    st.plotly_chart(pie_fig, use_container_width=True)

    # Time-series line plot: Sentiment over time, one line per keyword
    st.markdown("### 📈 Sentiment Score Over Time")
    line_df = filtered_df.copy()
    line_df = line_df.groupby(["Timestamp", "Keyword"]).agg({"Sentiment_Score": "mean"}).reset_index()

    line_fig = px.line(
        line_df,
        x="Timestamp",
        y="Sentiment_Score",
        color="Keyword",  # Split lines by keyword
        markers=True,
        title="Average Daily Sentiment Score by Keyword"
    )
    line_fig.update_layout(xaxis_title="Date", yaxis_title="Average Sentiment Score")
    st.plotly_chart(line_fig, use_container_width=True)

    # Top Positive and Negative Posts
    st.markdown("### 🔼 Top 5 Positive Posts")
    top_pos = filtered_df.nlargest(5, "Sentiment_Score")[["Sentiment_Score", "Original_Text"]]
    for i, row in top_pos.iterrows():
        st.success(f"{row['Sentiment_Score']:.2f}: {row['Original_Text'][:200]}")

    st.markdown("### 🔽 Top 5 Negative Posts")
    top_neg = filtered_df.nsmallest(5, "Sentiment_Score")[["Sentiment_Score", "Original_Text"]]
    for i, row in top_neg.iterrows():
        st.error(f"{row['Sentiment_Score']:.2f}: {row['Original_Text'][:200]}")

elif page == "2. Search Trends":
    st.title("📈 Search Trends Analysis")
    st.markdown("Analyzing search interest for selected memecoins over time.")

    trends_df = load_trends_data()
    sentiment_df = load_sentiment_data()

    # Sidebar keyword filter
    keywords = sorted(trends_df["Keyword"].unique())
    selected_keywords = st.multiselect("Select Keywords to Compare", keywords, default=keywords[:2])

    # Date range slider
    min_date = trends_df["Date"].min().date()
    max_date = trends_df["Date"].max().date()
    date_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    filtered_df = trends_df[
        (trends_df["Keyword"].isin(selected_keywords)) &
        (trends_df["Date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
    ]

    st.subheader("📈 Search Interest Over Time")

    # Rolling average toggle
    show_rolling = st.checkbox("Show 7-day Rolling Average", value=True)

    # Apply rolling average
    if show_rolling:
        smoothed_df = (
            filtered_df.sort_values("Date")
            .groupby("Keyword")
            .apply(lambda x: x.assign(Rolling_Score=x["Search_Score"].rolling(window=7, min_periods=1).mean()))
            .reset_index(drop=True)
        )
        y_col = "Rolling_Score"
    else:
        smoothed_df = filtered_df
        y_col = "Search_Score"

    # Line plot
    fig = px.line(
        smoothed_df,
        x="Date",
        y=y_col,
        color="Keyword",
        markers=True,
        title="Google Trends Search Score by Keyword"
    )
    fig.update_layout(xaxis_title="Date", yaxis_title=y_col.replace("_", " ").title())

    if not show_rolling:
        # Spike Detection & Annotation
        spikes = []
        for kw in selected_keywords:
            subset = smoothed_df[smoothed_df["Keyword"] == kw]
            threshold = np.percentile(subset["Search_Score"], 95)
            spike_points = subset[subset["Search_Score"] > threshold]
            spikes.append(spike_points)

        spike_df = pd.concat(spikes) if spikes else pd.DataFrame()

        for _, row in spike_df.iterrows():
            fig.add_annotation(
                x=row["Date"],
                y=row["Search_Score"],
                text=f"Spike: {row['Keyword']}",
                showarrow=True,
                arrowhead=1,
                yshift=10
            )
            # date_value = pd.Timestamp(row["Date"]).timestamp() * 1000 if isinstance(row["Date"], (pd.Timestamp, np.datetime64)) else row["Date"]
            # fig.add_vline(
            #     x=date_value,
            #     line_dash="solid",
            #     line_width=1,
            #     line_color="red",
            #     opacity=0.6,
            #     annotation=dict(
            #         # text=f"Spike: {row['Keyword']}",
            #         font=dict(color="red"),
            #         showarrow=True,
            #         arrowhead=2,
            #         ax=0,
            #         ay=-40
            #     ),
            #     # annotation_text=f"📈 {row['Keyword']}",
            #     annotation_position="top left"
            # )

    st.plotly_chart(fig, use_container_width=True)

elif page == "3. Price Correlation":
    st.title("🔗 Sentiment & Trend Correlation with Price")
    st.markdown("Correlating sentiment scores and Google Trends data with historical price movement.")
    # Correlation heatmap or scatter plots
    # Cross-correlation time-lag analysis
    # Include regression line or correlation coefficient

elif page == "4. Price Prediction":
    st.title("🔮 Forecasting Price with LSTM and HMM")
    st.markdown("Comparing model predictions for future price movement.")
    model_type = st.selectbox("Select Model", ["LSTM", "HMM"])
    # Line plot of actual vs predicted prices
    # Display evaluation metrics: RMSE, MAPE, etc.
    # Option to view prediction confidence intervals

elif page == "5. Memecoins vs Traditional Coins":
    st.title("⚖️ Memecoins vs Traditional Coins")
    st.markdown("Comparing price behavior and volatility between memecoins and established cryptocurrencies.")
    # Side-by-side line charts or candlestick charts
    # Volatility comparison (std dev)
    # Performance index or ROI

elif page == "6. Raw Data Explorer":
    st.title("🧾 Raw Data Viewer")
    st.markdown("View and filter all raw datasets used in this project.")
    # Tabs for:
    # - Sentiment Data
    # - Search Trend Data
    # - Price Data
    # Filtering options (date range, keyword, sentiment type)

