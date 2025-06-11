import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
import json
import ccxt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Load helper functions
from helper import load_trends_data, load_sentiment_data
from prediction_models import StockPricePredictor, create_sample_price_data, combine_all_data

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
    # start_date, end_date = st.date_input("Select Time Range", [pd.Timestamp("2025-04-01").date(), max_date], min_value=min_date, max_value=max_date)
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

    # Check if filtered DataFrame is empty
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

    print("$$%", filtered_df.describe())

    # Time-series line plot: Sentiment over time, one line per keyword
    st.markdown("### 📈 Sentiment Score Over Time (Rolling Averages)")
    line_df = filtered_df.copy()
    # line_df = line_df.groupby(["Timestamp", "Keyword"]).agg({"Sentiment_Score": "mean"}).reset_index()

    # remove outliers based on timestamp for cleaner visualization
    Q1 = line_df["Timestamp"].quantile(0.25)
    Q3 = line_df["Timestamp"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    line_df = line_df[
        (line_df["Timestamp"] >= lower_bound) &
        (line_df["Timestamp"] <= upper_bound)
    ]

    # Set desired height per row
    height_per_row = 300

    # Count number of rows needed for faceted line plot
    num_keywords = line_df["Keyword"].nunique()
    num_cols = 2
    num_rows = (num_keywords + num_cols - 1) // num_cols

    # Apply rolling average per keyword
    line_df.sort_values("Timestamp", inplace=True)
    line_df["Sentiment_Score_Smoothed"] = (
        line_df.groupby("Keyword")["Sentiment_Score"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    line_fig = px.line(
        line_df,
        x="Timestamp",
        y="Sentiment_Score_Smoothed",
        color="Keyword",  # Split lines by keyword
        facet_col="Keyword",
        facet_col_wrap=num_cols,
        markers=True,
        title="Average Daily Sentiment Score by Keyword"
    )

    line_fig.update_layout(
        # xaxis=dict(
        #     rangeselector=dict(
        #         buttons=list([
        #             dict(count=7, label="1w", step="day", stepmode="backward"),
        #             dict(count=1, label="1m", step="month", stepmode="backward"),
        #             dict(step="all")
        #         ])
        #     ),
        #     rangeslider=dict(visible=True),
        #     type="date"
        # ),
        height = num_rows * height_per_row,  # Set height based on number of rows
    )
    st.plotly_chart(line_fig, use_container_width=True)

    # Heatmap of Sentiment Scores
    heatmap_df = (
        filtered_df.groupby(["Timestamp", "Keyword"])["Sentiment_Score"]
        .mean().unstack()
    )

    st.markdown("### 🔥 Sentiment Heatmap")
    if not heatmap_df.empty:
        # Set up the figure
        plt.figure(figsize=(15, 10))  # adjust width/height as needed

        # Create heatmap
        ax = sns.heatmap(
            heatmap_df,
            cmap="RdYlGn",  # or 'vlag', 'coolwarm', etc.
            linecolor='gray',
            annot=False,  # You can turn this True to show values
            cbar_kws={'label': 'Avg Sentiment Score'},
            center=0
        )

        # Title and labels
        plt.title("🔥 Sentiment Heatmap by Keyword and Date", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Keyword")
        plt.xticks(rotation=45, ha='right')

        # Show in Streamlit
        st.pyplot(plt)
    else:
        st.warning("No data available for the heatmap.")
        st.stop()

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

    # Load data directly from data_prep
    from data_prep import prepare_data

    with st.spinner("Loading and preparing correlation data..."):
        try:
            final_data = prepare_data()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

    # Check if we have data
    if final_data.empty:
        st.error("No data available for correlation analysis.")
        st.stop()

    # Sidebar for coin selection
    st.sidebar.subheader("🔧 Correlation Configuration")
    available_coins = final_data['coin'].unique().tolist()
    selected_coin = st.sidebar.selectbox(
        "Select Coin for Analysis",
        options=available_coins,
        index=0 if 'pepe' in available_coins else 0
    )

    # Filter data for selected coin
    coin_data = final_data[final_data["coin"] == selected_coin].copy()

    if len(coin_data) == 0:
        st.error(f"No data available for {selected_coin}")
        st.stop()

    st.write(f"📈 **Analyzing {selected_coin.upper()}**: {len(coin_data)} records")

    # Show date range information
    if 'date' in coin_data.columns:
        min_date = coin_data['date'].min()
        max_date = coin_data['date'].max()
        st.write(f"📅 **Date Range**: {min_date} to {max_date}")

    # Check for NaN values and data quality
    st.subheader("📊 Data Quality Check")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_rows = len(coin_data)
        st.metric("Total Records", total_rows)

    with col2:
        valid_sentiment = coin_data['sentiment_mean'].notna().sum() if 'sentiment_mean' in coin_data.columns else 0
        st.metric("Valid Sentiment", f"{valid_sentiment}/{total_rows}")

    with col3:
        valid_trends = coin_data['Search_Score'].notna().sum() if 'Search_Score' in coin_data.columns else 0
        st.metric("Valid Search Trends", f"{valid_trends}/{total_rows}")

    with col4:
        valid_prices = coin_data['close'].notna().sum()
        st.metric("Valid Prices", f"{valid_prices}/{total_rows}")

    # Show data availability timeline
    st.subheader("📈 Data Availability Timeline")

    # Create availability visualization
    availability_df = coin_data[['date', 'close']].copy() if 'date' in coin_data.columns else coin_data[['close']].copy()

    if 'sentiment_mean' in coin_data.columns:
        availability_df['Has_Sentiment'] = coin_data['sentiment_mean'].notna()
    else:
        availability_df['Has_Sentiment'] = False

    if 'Search_Score' in coin_data.columns:
        availability_df['Has_Trends'] = coin_data['Search_Score'].notna()
    else:
        availability_df['Has_Trends'] = False

    fig_avail = go.Figure()

    if 'date' in availability_df.columns:
        fig_avail.add_trace(go.Scatter(
            x=availability_df['date'],
            y=availability_df['close'],
            mode='lines',
            name='Price',
            line=dict(color='blue')
        ))

        # Add markers for data availability
        sentiment_available = availability_df[availability_df['Has_Sentiment']]
        if not sentiment_available.empty:
            fig_avail.add_trace(go.Scatter(
                x=sentiment_available['date'],
                y=sentiment_available['close'],
                mode='markers',
                name='Has Sentiment Data',
                marker=dict(color='green', size=5, symbol='x')
            ))

        trends_available = availability_df[availability_df['Has_Trends']]
        if not trends_available.empty:
            fig_avail.add_trace(go.Scatter(
                x=trends_available['date'],
                y=trends_available['close'],
                mode='markers',
                name='Has Search Trends Data',
                marker=dict(color='red', size=3)
            ))

    fig_avail.update_layout(
        title=f"Data Availability Over Time for {selected_coin.upper()}",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    st.plotly_chart(fig_avail, use_container_width=True)

    # Prepare correlation data
    correlation_columns = ['close']
    if 'sentiment_mean' in coin_data.columns:
        correlation_columns.append('sentiment_mean')
    if 'positive_ratio' in coin_data.columns:
        correlation_columns.append('positive_ratio')
    if 'Search_Score' in coin_data.columns:
        correlation_columns.append('Search_Score')

    # Clean data for correlation analysis
    correlation_df = coin_data[correlation_columns].dropna()

    if len(correlation_df) > 0:
        # Calculate correlations
        correlation_data = correlation_df.corr()

        # Display correlation heatmap
        st.subheader("📊 Correlation Matrix")
        st.write(f"Correlation calculated on {len(correlation_df)} complete records")

        fig_corr = px.imshow(
            correlation_data,
            text_auto=True,
            aspect="auto",
            title=f"Correlation Matrix for {selected_coin.upper()}",
            color_continuous_scale="RdBu_r",
            labels={'color': 'Correlation'}
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Show correlation values
        st.subheader("📈 Correlation Values")

        correlation_metrics = []
        if 'sentiment_mean' in correlation_data.columns:
            price_sentiment_corr = correlation_data.loc['close', 'sentiment_mean']
            correlation_metrics.append(("Price vs Sentiment", price_sentiment_corr))

        if 'positive_ratio' in correlation_data.columns:
            price_positive_corr = correlation_data.loc['close', 'positive_ratio']
            correlation_metrics.append(("Price vs Positive Ratio", price_positive_corr))

        if 'Search_Score' in correlation_data.columns:
            price_trends_corr = correlation_data.loc['close', 'Search_Score']
            correlation_metrics.append(("Price vs Search Trends", price_trends_corr))

        # Display correlation metrics in columns
        if correlation_metrics:
            cols = st.columns(len(correlation_metrics))
            for i, (label, value) in enumerate(correlation_metrics):
                cols[i].metric(label, f"{value:.4f}")

        # Scatter plots with trendlines
        plot_columns = [col for col in correlation_columns if col != 'close']

        if len(plot_columns) >= 2:
            col1, col2 = st.columns(2)

            if 'sentiment_mean' in plot_columns:
                with col1:
                    st.subheader("💰 Price vs Sentiment")
                    if correlation_df['sentiment_mean'].nunique() > 1:
                        fig_scatter1 = px.scatter(
                            correlation_df,
                            x='sentiment_mean',
                            y='close',
                            trendline="ols",
                            title="Price vs Sentiment Score",
                            labels={'sentiment_mean': 'Sentiment Score', 'close': 'Price'}
                        )
                        st.plotly_chart(fig_scatter1, use_container_width=True)
                    else:
                        st.warning("Insufficient sentiment data variation for scatter plot")

            if 'Search_Score' in plot_columns:
                with col2:
                    st.subheader("🔍 Price vs Search Trends")
                    if correlation_df['Search_Score'].nunique() > 1:
                        fig_scatter2 = px.scatter(
                            correlation_df,
                            x='Search_Score',
                            y='close',
                            trendline="ols",
                            title="Price vs Search Score",
                            labels={'Search_Score': 'Search Score', 'close': 'Price'}
                        )
                        st.plotly_chart(fig_scatter2, use_container_width=True)
                    else:
                        st.warning("Insufficient search trends data variation for scatter plot")

        # Time series comparison with aligned data
        st.subheader("📈 Time Series Comparison")

        if 'date' in coin_data.columns and len(correlation_df) > 0:
            # Get dates for the correlation data
            correlation_with_dates = coin_data[correlation_columns + ['date']].dropna()

            if len(correlation_with_dates) > 0:
                # Normalize data for comparison
                normalized_df = correlation_with_dates.copy()
                for col in correlation_columns:
                    col_min = normalized_df[col].min()
                    col_max = normalized_df[col].max()
                    if col_max > col_min:
                        normalized_df[f'{col}_norm'] = (normalized_df[col] - col_min) / (col_max - col_min)
                    else:
                        normalized_df[f'{col}_norm'] = 0.5  # Set to middle if no variation

                fig_time = go.Figure()

                # Price line
                fig_time.add_trace(go.Scatter(
                    x=normalized_df['date'],
                    y=normalized_df['close_norm'],
                    mode='lines',
                    name='Price (Normalized)',
                    line=dict(color='blue', width=2)
                ))

                # Sentiment line
                if 'sentiment_mean' in correlation_columns:
                    fig_time.add_trace(go.Scatter(
                        x=normalized_df['date'],
                        y=normalized_df['sentiment_mean_norm'],
                        mode='lines',
                        name='Sentiment (Normalized)',
                        line=dict(color='green')
                    ))

                # Search trends line
                if 'Search_Score' in correlation_columns:
                    fig_time.add_trace(go.Scatter(
                        x=normalized_df['date'],
                        y=normalized_df['Search_Score_norm'],
                        mode='lines',
                        name='Search Trends (Normalized)',
                        line=dict(color='red')
                    ))

                # Positive ratio line
                if 'positive_ratio' in correlation_columns:
                    fig_time.add_trace(go.Scatter(
                        x=normalized_df['date'],
                        y=normalized_df['positive_ratio_norm'],
                        mode='lines',
                        name='Positive Ratio (Normalized)',
                        line=dict(color='orange')
                    ))

                fig_time.update_layout(
                    title=f"Normalized Time Series Comparison for {selected_coin.upper()} ({len(correlation_with_dates)} data points)",
                    xaxis_title="Date",
                    yaxis_title="Normalized Value",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_time, use_container_width=True)

        # Summary statistics
        st.subheader("📋 Summary Statistics")
        summary_stats = correlation_df.describe()
        st.dataframe(summary_stats, use_container_width=True)

    else:
        st.error("No complete data records found for correlation analysis")
        st.write("**Possible issues:**")
        st.write("- Missing sentiment or search trends data")
        st.write("- All values are NaN for key columns")
        st.write("- Data alignment issues")

        # Show debug information
        st.subheader("🔍 Debug Information")
        st.write("**Available columns:**", list(coin_data.columns))
        st.write("**Data types:**")
        for col in coin_data.columns:
            non_null_count = coin_data[col].notna().sum()
            st.write(f"- {col}: {non_null_count}/{len(coin_data)} non-null values")

        # Show sample of data
        st.write("**Sample of data:**")
        st.dataframe(coin_data.head(10))

elif page == "4. Price Prediction":
    st.title("🔮 Price Prediction Models")
    st.markdown("Train and evaluate LSTM, SVR, and Random Forest models for memecoin price prediction.")

    # Import required libraries for model training
    import joblib
    import os
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.impute import SimpleImputer

    # Try to import TensorFlow with fallback
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        st.warning("⚠️ TensorFlow not available. LSTM models will be disabled.")
        TENSORFLOW_AVAILABLE = False

    from data_prep import prepare_data

    # Get cleaned data
    with st.spinner("Loading and preparing data..."):
        try:
            final_data = prepare_data()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

    # Sidebar for model configuration
    st.sidebar.subheader("🔧 Model Configuration")

    # Coin selection
    available_coins = final_data['coin'].unique().tolist()
    selected_coin = st.sidebar.selectbox(
        "Select Memecoin",
        options=available_coins,
        index=0 if 'pepe' in available_coins else 0
    )

    # Model selection - exclude LSTM if TensorFlow not available
    model_options = ["Random Forest", "SVR"]
    if TENSORFLOW_AVAILABLE:
        model_options.insert(0, "LSTM")
    model_options.append("All Models")

    model_type = st.sidebar.selectbox(
        "Select Model Type",
        options=model_options
    )

    # Training options
    use_pretrained = st.sidebar.checkbox("Use Pre-trained Models", value=True)

    if not use_pretrained:
        train_models = st.sidebar.button("🚀 Train Models")
    else:
        train_models = False

    # Prediction options
    st.sidebar.subheader("📊 Prediction Options")
    prediction_days = st.sidebar.slider("Days to Predict", 1, 30, 7)

    # Filter data for selected coin
    coin_data = final_data[final_data["coin"] == selected_coin].copy()

    # Preserve date information before dropping columns
    date_info = coin_data[['date', 'timestamp']].copy() if 'date' in coin_data.columns else None

    coin_data = coin_data.drop(columns=["timestamp", "date", "coin"], errors='ignore')
    coin_data = coin_data.dropna()

    # Align date_info with cleaned coin_data
    if date_info is not None:
        date_info = date_info.iloc[:len(coin_data)].reset_index(drop=True)

    if len(coin_data) == 0:
        st.error(f"No data available for {selected_coin}")
        st.stop()

    st.write(f"📈 **Data for {selected_coin.upper()}**: {len(coin_data)} records")

    # Show date range information
    if date_info is not None and 'date' in date_info.columns:
        min_date = date_info['date'].min()
        max_date = date_info['date'].max()
        st.write(f"📅 **Date Range**: {min_date} to {max_date}")

    # Prepare features and target
    target_columns = ['close', 'target', 'target_change']
    feature_columns = [col for col in coin_data.columns if col not in target_columns]

    if len(feature_columns) == 0:
        st.error("No feature columns available for training")
        st.stop()

    X = coin_data[feature_columns]
    y = coin_data['close']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Training", "📈 Model Performance", "🔮 Price Prediction", "📉 Feature Analysis"])

    with tab1:
        st.subheader("🤖 Model Training")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Dataset Information:**")
            st.write(f"- Total samples: {len(coin_data)}")
            st.write(f"- Features: {len(feature_columns)}")
            st.write(f"- Training samples: {len(X_train)}")
            st.write(f"- Test samples: {len(X_test)}")

        with col2:
            st.write("**Feature columns:**")
            for col in feature_columns[:10]:  # Show first 10 features
                st.write(f"- {col}")
            if len(feature_columns) > 10:
                st.write(f"... and {len(feature_columns) - 10} more")

        # Model training section
        if train_models or not use_pretrained:
            st.write("---")
            st.subheader("🚀 Training Models")

            # Initialize containers for results
            models = {}
            scalers = {}
            metrics = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                if model_type in ["LSTM", "All Models"] and TENSORFLOW_AVAILABLE:
                    status_text.text("Training LSTM Model...")
                    progress_bar.progress(10)

                    # Prepare data for LSTM - exactly like memecoin_mania.py
                    scaler = MinMaxScaler()

                    # Scale only the close price (target variable) - exact implementation
                    scaled_close = scaler.fit_transform(y_train.values.reshape(-1, 1))

                    # Create sequences for LSTM model - exact implementation
                    sequence_length = 50
                    X_lstm = []
                    y_lstm = []

                    for i in range(sequence_length, len(scaled_close)):
                        X_lstm.append(scaled_close[i-sequence_length:i, 0])  # Extract column 0
                        y_lstm.append(scaled_close[i, 0])  # Extract scalar value

                    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
                    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

                    # Check if we have enough data
                    if len(X_lstm) == 0:
                        st.error(f"Not enough data for LSTM training. Need at least {sequence_length + 1} samples, got {len(y_train)}")
                        progress_bar.progress(40)
                    else:
                        # Build LSTM model - exact implementation from memecoin_mania.py
                        model = Sequential()
                        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
                        model.add(Dropout(0.2))
                        model.add(LSTM(units=50, return_sequences=False))
                        model.add(Dropout(0.2))
                        model.add(Dense(units=1))  # Prediction of the next price

                        model.compile(optimizer='adam', loss='mean_squared_error')

                        # Training the model - exact implementation
                        with st.spinner("Training LSTM..."):
                            model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)  # Reduced epochs for demo

                        # Save model and scaler
                        model.save(f'lstm_model_{selected_coin}.h5')
                        joblib.dump(scaler, f'scaler_lstm_{selected_coin}.pkl')

                        models['LSTM'] = model
                        scalers['LSTM'] = scaler

                        # Calculate metrics on test data using exact approach
                        if len(y_test) > sequence_length:
                            # Scale test target
                            scaled_test_close = scaler.transform(y_test.values.reshape(-1, 1))

                            # Create test sequences - exact implementation
                            X_test_lstm = []
                            y_test_lstm = []

                            for i in range(sequence_length, len(scaled_test_close)):
                                X_test_lstm.append(scaled_test_close[i-sequence_length:i, 0])
                                y_test_lstm.append(scaled_test_close[i, 0])

                            if X_test_lstm:
                                X_test_lstm, y_test_lstm = np.array(X_test_lstm), np.array(y_test_lstm)
                                X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], sequence_length, 1))

                                # Make predictions - exact implementation
                                predictions = []
                                for i in range(len(X_test_lstm)):
                                    test_sequence = X_test_lstm[i].reshape(1, sequence_length, 1)
                                    prediction = model.predict(test_sequence, verbose=0)
                                    predictions.append(scaler.inverse_transform(prediction)[0][0])

                                # Convert y_test_lstm back to original scale for metrics
                                y_test_actual = scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()

                                lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
                                lstm_mae = mean_absolute_error(y_test_actual, predictions)
                                lstm_r2 = r2_score(y_test_actual, predictions)

                                metrics['LSTM'] = {
                                    'RMSE': lstm_rmse,
                                    'MAE': lstm_mae,
                                    'R²': lstm_r2
                                }

                    progress_bar.progress(40)

                if model_type in ["Random Forest", "All Models"]:
                    status_text.text("Training Random Forest Model...")

                    # Handle missing values
                    imputer = SimpleImputer(strategy='mean')
                    X_train_imputed = imputer.fit_transform(X_train)
                    X_test_imputed = imputer.transform(X_test)

                    # Random Forest Model
                    with st.spinner("Training Random Forest..."):
                        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                        rf_model.fit(X_train_imputed, y_train)

                    # Save model
                    joblib.dump(rf_model, f'rf_model_{selected_coin}.pkl')
                    joblib.dump(imputer, f'imputer_rf_{selected_coin}.pkl')

                    models['Random Forest'] = rf_model
                    scalers['Random Forest'] = imputer

                    # Calculate metrics
                    rf_pred = rf_model.predict(X_test_imputed)
                    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
                    rf_mae = mean_absolute_error(y_test, rf_pred)
                    rf_r2 = r2_score(y_test, rf_pred)

                    metrics['Random Forest'] = {
                        'RMSE': rf_rmse,
                        'MAE': rf_mae,
                        'R²': rf_r2
                    }

                    progress_bar.progress(70)

                if model_type in ["SVR", "All Models"]:
                    status_text.text("Training SVR Model...")

                    try:
                        # Prepare data for SVR with better preprocessing for small values
                        imputer = SimpleImputer(strategy='mean')

                        # Handle small price values by scaling appropriately
                        price_scale_factor = 1.0
                        if y_train.max() < 0.01:  # For coins like PEPE, SHIBA
                            price_scale_factor = 1000000  # Scale up small values
                            y_train_scaled = y_train * price_scale_factor
                            y_test_scaled = y_test * price_scale_factor
                        else:
                            y_train_scaled = y_train
                            y_test_scaled = y_test

                        # Standard feature scaling
                        scaler_svr = StandardScaler()

                        X_train_imputed = imputer.fit_transform(X_train)
                        X_train_scaled = scaler_svr.fit_transform(X_train_imputed)

                        X_test_imputed = imputer.transform(X_test)
                        X_test_scaled = scaler_svr.transform(X_test_imputed)

                        # SVR Model with parameters better suited for small values
                        svr_model = SVR(
                            kernel='rbf',
                            C=1000,  # Increased C for better fitting
                            gamma='scale',  # Let sklearn determine gamma
                            epsilon=0.0001  # Smaller epsilon for small values
                        )

                        with st.spinner("Training SVR..."):
                            svr_model.fit(X_train_scaled, y_train_scaled)

                        # Test predictions
                        svr_pred_scaled = svr_model.predict(X_test_scaled)
                        svr_pred = svr_pred_scaled / price_scale_factor  # Scale back down

                        # Calculate metrics
                        svr_rmse = np.sqrt(mean_squared_error(y_test, svr_pred))
                        svr_mae = mean_absolute_error(y_test, svr_pred)
                        svr_r2 = r2_score(y_test, svr_pred)

                        # Save model with scale factor
                        joblib.dump(svr_model, f'svr_model_{selected_coin}.pkl')
                        joblib.dump(scaler_svr, f'scaler_svr_{selected_coin}.pkl')
                        joblib.dump(imputer, f'imputer_svr_{selected_coin}.pkl')
                        joblib.dump(price_scale_factor, f'price_scale_factor_{selected_coin}.pkl')

                        models['SVR'] = svr_model
                        scalers['SVR'] = (scaler_svr, imputer, price_scale_factor)

                        metrics['SVR'] = {
                            'RMSE': svr_rmse,
                            'MAE': svr_mae,
                            'R²': svr_r2
                        }

                        st.success(f"✅ SVR model trained successfully for {selected_coin}")

                    except Exception as e:
                        st.warning(f"⚠️ SVR training failed for {selected_coin}: {str(e)}")
                        st.info("This is common for coins with very small price values. Other models will still work.")

                    progress_bar.progress(100)

                status_text.text("✅ Training completed!")

                # Display training results
                if metrics:
                    st.success("🎉 Models trained successfully!")

                    metrics_df = pd.DataFrame(metrics).T
                    st.write("**Training Metrics:**")
                    st.dataframe(metrics_df.round(4))

            except Exception as e:
                st.error(f"Error during training: {str(e)}")

        else:
            st.info("📁 Using pre-trained models. Toggle 'Use Pre-trained Models' to train new models.")

    with tab2:
        st.subheader("📊 Model Performance Comparison")

        # Load or use trained models
        available_models = []
        model_predictions = {}

        # Check for LSTM model
        if TENSORFLOW_AVAILABLE and os.path.exists(f'lstm_model_{selected_coin}.h5') and os.path.exists(f'scaler_lstm_{selected_coin}.pkl'):
            available_models.append('LSTM')

        # Check for Random Forest model
        if os.path.exists(f'rf_model_{selected_coin}.pkl') and os.path.exists(f'imputer_rf_{selected_coin}.pkl'):
            available_models.append('Random Forest')

        # Check for SVR model
        if os.path.exists(f'svr_model_{selected_coin}.pkl'):
            available_models.append('SVR')

        if not available_models:
            st.warning("⚠️ No trained models found. Please train models first.")
        else:
            st.write(f"📈 **Available Models**: {', '.join(available_models)}")

            # Load models and make predictions
            for model_name in available_models:
                if model_name == 'LSTM' and TENSORFLOW_AVAILABLE:
                    model = load_model(f'lstm_model_{selected_coin}.h5')
                    scaler = joblib.load(f'scaler_lstm_{selected_coin}.pkl')

                    # Prepare test data for LSTM - exact implementation
                    scaled_test_close = scaler.transform(y_test.values.reshape(-1, 1))
                    sequence_length = 50

                    if len(scaled_test_close) > sequence_length:
                        X_test_lstm = []
                        y_test_lstm = []

                        for i in range(sequence_length, len(scaled_test_close)):
                            X_test_lstm.append(scaled_test_close[i-sequence_length:i, 0])
                            y_test_lstm.append(scaled_test_close[i, 0])

                        if X_test_lstm:
                            X_test_lstm = np.array(X_test_lstm)
                            X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], sequence_length, 1))

                            # Make predictions - exact implementation
                            predictions = []
                            for i in range(len(X_test_lstm)):
                                test_sequence = X_test_lstm[i].reshape(1, sequence_length, 1)
                                prediction = model.predict(test_sequence, verbose=0)
                                predictions.append(scaler.inverse_transform(prediction)[0][0])

                            model_predictions['LSTM'] = predictions

                elif model_name == 'Random Forest':
                    rf_model = joblib.load(f'rf_model_{selected_coin}.pkl')
                    imputer_rf = joblib.load(f'imputer_rf_{selected_coin}.pkl')

                    X_test_imputed = imputer_rf.transform(X_test)
                    rf_pred = rf_model.predict(X_test_imputed)
                    model_predictions['Random Forest'] = rf_pred

                elif model_name == 'SVR':
                    try:
                        svr_model = joblib.load(f'svr_model_{selected_coin}.pkl')
                        scaler_svr = joblib.load(f'scaler_svr_{selected_coin}.pkl')
                        imputer_svr = joblib.load(f'imputer_svr_{selected_coin}.pkl')

                        # Try to load price scale factor, default to 1.0 if not found
                        try:
                            price_scale_factor = joblib.load(f'price_scale_factor_{selected_coin}.pkl')
                        except:
                            price_scale_factor = 1.0

                        X_test_imputed = imputer_svr.transform(X_test)
                        X_test_scaled = scaler_svr.transform(X_test_imputed)
                        svr_pred_scaled = svr_model.predict(X_test_scaled)
                        svr_pred = svr_pred_scaled / price_scale_factor  # Scale back down
                        model_predictions['SVR'] = svr_pred

                    except Exception as e:
                        st.warning(f"⚠️ Error loading SVR model for {selected_coin}: {str(e)}")

            # Display performance metrics and visualizations
            if model_predictions:
                # Calculate metrics for all models
                performance_metrics = {}

                for model_name, predictions in model_predictions.items():
                    # Align predictions with actual values
                    y_actual = y_test.values[-len(predictions):]

                    if len(y_actual) > 0:
                        rmse = np.sqrt(mean_squared_error(y_actual, predictions))
                        mae = mean_absolute_error(y_actual, predictions)
                        r2 = r2_score(y_actual, predictions)

                        performance_metrics[model_name] = {
                            'RMSE': rmse,
                            'MAE': mae,
                            'R²': r2
                        }

                if performance_metrics:
                    # Display metrics table
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Performance Metrics:**")
                        metrics_df = pd.DataFrame(performance_metrics).T
                        st.dataframe(metrics_df.round(4))

                    with col2:
                        # Best model
                        best_model = min(performance_metrics.keys(),
                                       key=lambda x: performance_metrics[x]['RMSE'])
                        st.metric("🏆 Best Model (Lowest RMSE)", best_model)
                        st.metric("RMSE", f"{performance_metrics[best_model]['RMSE']:.4f}")
                        st.metric("R² Score", f"{performance_metrics[best_model]['R²']:.4f}")

                    # Visualization: Actual vs Predicted
                    fig = go.Figure()

                    # Add actual prices
                    y_actual = y_test.values
                    test_dates = list(range(len(y_actual)))  # Convert range to list

                    fig.add_trace(go.Scatter(
                        x=test_dates,
                        y=y_actual,
                        mode='lines+markers',
                        name='Actual Price',
                        line=dict(color='black', width=2)
                    ))

                    # Add predictions for each model
                    colors = ['red', 'blue', 'green', 'orange', 'purple']
                    for i, (model_name, predictions) in enumerate(model_predictions.items()):
                        pred_dates = list(range(len(y_actual) - len(predictions), len(y_actual)))  # Convert range to list

                        fig.add_trace(go.Scatter(
                            x=pred_dates,
                            y=predictions,
                            mode='lines+markers',
                            name=f'{model_name} Prediction',
                            line=dict(color=colors[i % len(colors)])
                        ))

                    fig.update_layout(
                        title=f'Model Predictions vs Actual Prices for {selected_coin.upper()}',
                        xaxis_title='Time (Test Period)',
                        yaxis_title='Price',
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Model comparison bar chart
                    fig_bar = go.Figure()

                    models = list(performance_metrics.keys())
                    rmse_values = [performance_metrics[model]['RMSE'] for model in models]
                    r2_values = [performance_metrics[model]['R²'] for model in models]

                    fig_bar.add_trace(go.Bar(
                        x=models,
                        y=rmse_values,
                        name='RMSE',
                        yaxis='y',
                        offsetgroup=1
                    ))

                    fig_bar.add_trace(go.Bar(
                        x=models,
                        y=r2_values,
                        name='R² Score',
                        yaxis='y2',
                        offsetgroup=2
                    ))

                    fig_bar.update_layout(
                        title='Model Performance Comparison',
                        xaxis_title='Models',
                        yaxis=dict(title='RMSE', side='left'),
                        yaxis2=dict(title='R² Score', side='right', overlaying='y'),
                        barmode='group'
                    )

                    st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("🔮 Future Price Prediction")

        if not available_models:
            st.warning("⚠️ No trained models available for prediction.")
        else:
            # Select model for prediction
            prediction_model = st.selectbox(
                "Choose Model for Prediction",
                options=available_models
            )

            if st.button("🚀 Generate Predictions"):
                try:
                    # Get latest data for prediction
                    latest_data = coin_data.tail(prediction_days + 50)  # Get more data for LSTM sequences

                    future_predictions = []

                    if prediction_model == 'LSTM' and TENSORFLOW_AVAILABLE:
                        model = load_model(f'lstm_model_{selected_coin}.h5')
                        scaler = joblib.load(f'scaler_lstm_{selected_coin}.pkl')

                        # Get the most recent data for prediction - exact implementation
                        sequence_length = 50

                        if len(coin_data) < sequence_length:
                            st.error(f"Not enough data for LSTM prediction. Need at least {sequence_length} records, got {len(coin_data)}")
                        else:
                            # Extract last sequence_length rows - exact implementation
                            recent_close_data = coin_data['close'].tail(sequence_length).values
                            scaled_recent = scaler.transform(recent_close_data.reshape(-1, 1))

                            # Extract only the column used for training - exact implementation
                            test_data = scaled_recent[-sequence_length:, 0]
                            # Reshape into (1, 50, 1) - exact implementation
                            test_data = test_data.reshape(1, sequence_length, 1).astype(np.float32)

                            for day in range(prediction_days):
                                # Predict the next price - exact implementation
                                predicted_price_scaled = model.predict(test_data, verbose=0)
                                predicted_price = scaler.inverse_transform(predicted_price_scaled)  # Undo scaling
                                future_predictions.append(predicted_price[0][0])

                                # Update sequence for next prediction
                                # Add the new prediction to the sequence and remove the oldest
                                new_scaled_value = predicted_price_scaled[0][0]
                                test_data = np.roll(test_data, -1, axis=1)
                                test_data[0, -1, 0] = new_scaled_value

                    elif prediction_model == 'Random Forest':
                        rf_model = joblib.load(f'rf_model_{selected_coin}.pkl')
                        imputer_rf = joblib.load(f'imputer_rf_{selected_coin}.pkl')

                        # Get the most recent features
                        latest_features = latest_data[feature_columns].tail(1).copy()

                        for day in range(prediction_days):
                            # Impute and predict
                            latest_features_imputed = imputer_rf.transform(latest_features)
                            pred = rf_model.predict(latest_features_imputed)[0]
                            future_predictions.append(pred)

                            # Conservative feature updating - only update price-related features
                            if day < prediction_days - 1:  # Don't update on last iteration
                                # Update only close price and price change
                                if 'close' in latest_features.columns:
                                    latest_features.loc[:, 'close'] = pred

                                # Update price change more conservatively
                                if 'price_change' in latest_features.columns:
                                    if len(future_predictions) > 1:
                                        price_change = (pred - future_predictions[-2]) / abs(future_predictions[-2])
                                        # Limit extreme changes to prevent instability
                                        price_change = np.clip(price_change, -0.1, 0.1)
                                        latest_features.loc[:, 'price_change'] = price_change

                                # Update moving averages more smoothly
                                if 'sma_5' in latest_features.columns:
                                    current_sma5 = latest_features.loc[:, 'sma_5'].iloc[0]
                                    # Smooth update: 80% old value + 20% new price
                                    latest_features.loc[:, 'sma_5'] = 0.8 * current_sma5 + 0.2 * pred

                                if 'sma_20' in latest_features.columns:
                                    current_sma20 = latest_features.loc[:, 'sma_20'].iloc[0]
                                    # Even smoother update for longer MA
                                    latest_features.loc[:, 'sma_20'] = 0.95 * current_sma20 + 0.05 * pred

                                # Keep other features (sentiment, trends, volume) relatively stable
                                # Apply small random noise to prevent overfitting to static values
                                for col in latest_features.columns:
                                    if col not in ['close', 'price_change', 'sma_5', 'sma_20']:
                                        current_val = latest_features.loc[:, col].iloc[0]
                                        # Add small random variation (±1%)
                                        noise = np.random.normal(0, 0.01) * abs(current_val) if current_val != 0 else 0
                                        latest_features.loc[:, col] = current_val + noise

                    elif prediction_model == 'SVR':
                        try:
                            svr_model = joblib.load(f'svr_model_{selected_coin}.pkl')
                            scaler_svr = joblib.load(f'scaler_svr_{selected_coin}.pkl')
                            imputer_svr = joblib.load(f'imputer_svr_{selected_coin}.pkl')

                            # Try to load price scale factor, default to 1.0 if not found
                            try:
                                price_scale_factor = joblib.load(f'price_scale_factor_{selected_coin}.pkl')
                            except:
                                price_scale_factor = 1.0

                            # Get the most recent features
                            latest_features = latest_data[feature_columns].tail(1).copy()

                            for day in range(prediction_days):
                                # Impute, scale and predict
                                latest_features_imputed = imputer_svr.transform(latest_features)
                                latest_features_scaled = scaler_svr.transform(latest_features_imputed)
                                pred_scaled = svr_model.predict(latest_features_scaled)[0]
                                pred = pred_scaled / price_scale_factor  # Scale back down
                                future_predictions.append(pred)

                                # Conservative feature updating for SVR
                                if day < prediction_days - 1:  # Don't update on last iteration
                                    # Update only close price and price change
                                    if 'close' in latest_features.columns:
                                        latest_features.loc[:, 'close'] = pred

                                    # Update price change more conservatively
                                    if 'price_change' in latest_features.columns:
                                        if len(future_predictions) > 1:
                                            price_change = (pred - future_predictions[-2]) / abs(future_predictions[-2])
                                            # Limit extreme changes to prevent instability
                                            price_change = np.clip(price_change, -0.1, 0.1)
                                            latest_features.loc[:, 'price_change'] = price_change

                                    # Update moving averages more smoothly
                                    if 'sma_5' in latest_features.columns:
                                        current_sma5 = latest_features.loc[:, 'sma_5'].iloc[0]
                                        # Smooth update: 80% old value + 20% new price
                                        latest_features.loc[:, 'sma_5'] = 0.8 * current_sma5 + 0.2 * pred

                                    if 'sma_20' in latest_features.columns:
                                        current_sma20 = latest_features.loc[:, 'sma_20'].iloc[0]
                                        # Even smoother update for longer MA
                                        latest_features.loc[:, 'sma_20'] = 0.95 * current_sma20 + 0.05 * pred

                                    # Keep other features stable with minimal noise
                                    for col in latest_features.columns:
                                        if col not in ['close', 'price_change', 'sma_5', 'sma_20']:
                                            current_val = latest_features.loc[:, col].iloc[0]
                                            # Add very small random variation (±0.5%)
                                            noise = np.random.normal(0, 0.005) * abs(current_val) if current_val != 0 else 0
                                            latest_features.loc[:, col] = current_val + noise

                        except Exception as e:
                            st.error(f"Error with SVR prediction for {selected_coin}: {str(e)}")
                            st.info("SVR may not work well with very small price values. Try LSTM or Random Forest.")
                            # break  # Exit the prediction loop on error

                    if future_predictions:
                        # Create prediction visualization with actual dates
                        historical_prices = coin_data['close'].tail(30).values

                        # Use actual dates if available, otherwise use indices
                        if date_info is not None and 'date' in date_info.columns:
                            # Get the last 30 historical dates
                            historical_dates_actual = date_info['date'].tail(30).tolist()

                            # Generate future dates based on the last historical date
                            last_date = pd.to_datetime(historical_dates_actual[-1])
                            future_dates_actual = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d')
                                                 for i in range(prediction_days)]

                            # Convert historical dates to strings for consistency
                            historical_dates_str = [pd.to_datetime(d).strftime('%Y-%m-%d')
                                                  for d in historical_dates_actual]

                            fig_pred = go.Figure()

                            # Historical prices with actual dates
                            fig_pred.add_trace(go.Scatter(
                                x=historical_dates_str,
                                y=historical_prices,
                                mode='lines+markers',
                                name='Historical Prices',
                                line=dict(color='blue')
                            ))

                            # Future predictions with actual dates
                            fig_pred.add_trace(go.Scatter(
                                x=future_dates_actual,
                                y=future_predictions,
                                mode='lines+markers',
                                name=f'{prediction_model} Predictions',
                                line=dict(color='red', dash='dash')
                            ))

                            # Connection point
                            fig_pred.add_trace(go.Scatter(
                                x=[historical_dates_str[-1]],
                                y=[historical_prices[-1]],
                                mode='markers',
                                name='Current Price',
                                marker=dict(color='green', size=10)
                            ))

                            fig_pred.update_layout(
                                title=f'{prediction_days}-Day Price Prediction for {selected_coin.upper()} using {prediction_model}',
                                xaxis_title='Date',
                                yaxis_title='Price',
                                hovermode='x unified',
                                xaxis=dict(
                                    tickangle=45,
                                    tickformat='%Y-%m-%d'
                                )
                            )

                            # Update prediction table with actual dates
                            pred_df = pd.DataFrame({
                                'Date': future_dates_actual,
                                'Predicted Price': [f"${p:.6f}" for p in future_predictions],
                                'Daily Change': [f"{((future_predictions[i] - (historical_prices[-1] if i == 0 else future_predictions[i-1])) / (historical_prices[-1] if i == 0 else future_predictions[i-1]) * 100):.2f}%" for i in range(prediction_days)]
                            })

                        else:
                            # Fallback to generic day indices if no date info
                            historical_dates = list(range(-30, 0))
                            future_dates = list(range(1, prediction_days + 1))

                            fig_pred = go.Figure()

                            # Historical prices
                            fig_pred.add_trace(go.Scatter(
                                x=historical_dates,
                                y=historical_prices,
                                mode='lines+markers',
                                name='Historical Prices',
                                line=dict(color='blue')
                            ))

                            # Future predictions
                            fig_pred.add_trace(go.Scatter(
                                x=future_dates,
                                y=future_predictions,
                                mode='lines+markers',
                                name=f'{prediction_model} Predictions',
                                line=dict(color='red', dash='dash')
                            ))

                            # Connection point
                            fig_pred.add_trace(go.Scatter(
                                x=[0],
                                y=[historical_prices[-1]],
                                mode='markers',
                                name='Current Price',
                                marker=dict(color='green', size=10)
                            ))

                            fig_pred.update_layout(
                                title=f'{prediction_days}-Day Price Prediction for {selected_coin.upper()} using {prediction_model}',
                                xaxis_title='Days (Negative = Historical, Positive = Future)',
                                yaxis_title='Price',
                                hovermode='x unified'
                            )

                            # Generic prediction table
                            pred_df = pd.DataFrame({
                                'Day': list(range(1, prediction_days + 1)),
                                'Predicted Price': [f"${p:.6f}" for p in future_predictions],
                                'Daily Change': [f"{((future_predictions[i] - (historical_prices[-1] if i == 0 else future_predictions[i-1])) / (historical_prices[-1] if i == 0 else future_predictions[i-1]) * 100):.2f}%" for i in range(prediction_days)]
                            })

                        st.plotly_chart(fig_pred, use_container_width=True)

                        # Prediction summary
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            current_price = historical_prices[-1]
                            st.metric("Current Price", f"${current_price:.6f}")

                        with col2:
                            predicted_price = future_predictions[-1]
                            price_change = predicted_price - current_price
                            st.metric(
                                f"Predicted Price ({prediction_days}d)",
                                f"${predicted_price:.6f}",
                                f"${price_change:.6f}"
                            )

                        with col3:
                            change_percent = (price_change / current_price) * 100
                            st.metric(
                                "Predicted Change",
                                f"{change_percent:.2f}%",
                                f"{'📈' if change_percent > 0 else '📉'}"
                            )

                        st.write("**Detailed Predictions:**")
                        st.dataframe(pred_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
                    st.write("Please check if models are properly trained and saved.")
                    # Show debug information
                    st.write("**Debug Information:**")
                    st.write(f"- Selected model: {prediction_model}")
                    st.write(f"- Coin data length: {len(coin_data)}")
                    st.write(f"- Feature columns: {len(feature_columns)}")
                    st.write(f"- Available models on disk: {available_models}")

    with tab4:
        st.subheader("📊 Feature Importance Analysis")

        if 'Random Forest' in available_models:
            try:
                rf_model = joblib.load(f'rf_model_{selected_coin}.pkl')

                # Get feature importance
                importance_scores = rf_model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': importance_scores
                }).sort_values('Importance', ascending=False)

                # Feature importance chart
                fig_importance = px.bar(
                    feature_importance_df.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f'Top 15 Feature Importance for {selected_coin.upper()} (Random Forest)'
                )

                fig_importance.update_layout(height=600)
                st.plotly_chart(fig_importance, use_container_width=True)

                # Feature importance table
                st.write("**All Features Ranked by Importance:**")
                st.dataframe(feature_importance_df, use_container_width=True)

                # Feature correlation with target
                st.subheader("📈 Feature Correlation with Price")

                correlation_data = coin_data[feature_columns + ['close']].corr()['close'].abs().sort_values(ascending=False)
                correlation_df = pd.DataFrame({
                    'Feature': correlation_data.index[1:],  # Exclude self-correlation
                    'Correlation': correlation_data.values[1:]
                })

                fig_corr = px.bar(
                    correlation_df.head(15),
                    x='Correlation',
                    y='Feature',
                    orientation='h',
                    title='Top 15 Features by Correlation with Price'
                )

                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, use_container_width=True)

            except Exception as e:
                st.error(f"Error loading Random Forest model for feature analysis: {str(e)}")
        else:
            st.info("Feature importance analysis requires a trained Random Forest model.")

            # Show basic correlation analysis
            if len(feature_columns) > 0:
                st.subheader("📈 Basic Feature Correlation with Price")

                correlation_data = coin_data[feature_columns + ['close']].corr()['close'].abs().sort_values(ascending=False)
                correlation_df = pd.DataFrame({
                    'Feature': correlation_data.index[1:],  # Exclude self-correlation
                    'Correlation': correlation_data.values[1:]
                })
                st.dataframe(correlation_df.head(15), use_container_width=True)

elif page == "5. Memecoins vs Traditional Coins":
    st.title("⚖️ Memecoins vs Traditional Coins")
    st.markdown("Comparing price behavior and volatility between memecoins and established cryptocurrencies.")

    # Load data
    sentiment_df = load_sentiment_data()
    trends_df = load_trends_data()
    memecoin_price_df = create_sample_price_data(sentiment_df, trends_df)

    # Create traditional coin data (Bitcoin simulation)
    btc_dates = pd.date_range(
        start=memecoin_price_df["Date"].min(),
        end=memecoin_price_df["Date"].max(),
        freq="D",
    )

    np.random.seed(123)
    btc_prices = []
    current_btc = 45000
    for _ in btc_dates:
        change = np.random.normal(0, 0.03)  # Lower volatility for BTC
        current_btc = max(1000, current_btc * (1 + change))
        btc_prices.append(current_btc)

    btc_df = pd.DataFrame(
        {"Date": btc_dates, "Close": btc_prices, "Type": "Traditional (BTC)"}
    )

    memecoin_compare_df = memecoin_price_df[["Date", "Close"]].copy()
    memecoin_compare_df["Type"] = "Memecoin (DOGE)"

    # Normalize prices for comparison
    memecoin_compare_df["Normalized_Price"] = (
        memecoin_compare_df["Close"] / memecoin_compare_df["Close"].iloc[0]
    ) * 100

    btc_df["Normalized_Price"] = (btc_df["Close"] / btc_df["Close"].iloc[0]) * 100
    compare_df = pd.concat([memecoin_compare_df, btc_df])

    # Price comparison
    st.subheader("📈 Normalized Price Comparison")
    fig_compare = px.line(
        compare_df,
        x="Date",
        y="Normalized_Price",
        color="Type",
        title="Memecoin vs Traditional Crypto Performance (Base 100)",
    )

    st.plotly_chart(fig_compare, use_container_width=True)

    # Volatility analysis
    st.subheader("📊 Volatility Analysis")
    memecoin_volatility = memecoin_compare_df["Close"].pct_change().std() * 100
    btc_volatility = btc_df["Close"].pct_change().std() * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Memecoin Volatility", f"{memecoin_volatility:.2f}%")

    with col2:
        st.metric("Bitcoin Volatility", f"{btc_volatility:.2f}%")

    with col3:
        volatility_ratio = memecoin_volatility / btc_volatility
        st.metric("Volatility Ratio", f"{volatility_ratio:.2f}x")

    # Distribution of returns
    st.subheader("📊 Return Distribution")
    memecoin_returns = memecoin_compare_df["Close"].pct_change().dropna() * 100
    btc_returns = btc_df["Close"].pct_change().dropna() * 100

    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Histogram(
            x=memecoin_returns, name="Memecoin Returns", opacity=0.7, nbinsx=30
        )
    )
    fig_dist.add_trace(
        go.Histogram(x=btc_returns, name="Bitcoin Returns", opacity=0.7, nbinsx=30)
    )

    fig_dist.update_layout(
        title="Distribution of Daily Returns (%)",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        barmode="overlay",
    )

    st.plotly_chart(fig_dist, use_container_width=True)

elif page == "6. Raw Data Explorer":
    st.title("🧪 Raw Data Explorer")

    # Load datasets
    sentiment_df = load_sentiment_data()
    trend_df = load_trends_data()

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Sentiment Data", "📈 Trend Data", "💰 Price Data"])

    # ------------------- #
    # Sentiment Tab
    # ------------------- #
    with tab1:
        st.subheader("Sentiment Data Explorer")

        # Filters
        keywords = sorted(sentiment_df["Keyword"].unique())
        sentiment_labels = sorted(sentiment_df["Label"].unique())
        min_date, max_date = sentiment_df["Timestamp"].min(), sentiment_df["Timestamp"].max()

        selected_keywords = st.multiselect("Select Keywords", keywords, default=keywords)
        selected_labels = st.multiselect("Select Sentiment Labels", sentiment_labels, default=sentiment_labels)
        start_date, end_date = st.date_input("Select Time Range", [min_date.date(), max_date.date()])

        # date_range = st.slider(
        #     "Select Date Range",
        #     min_value=min_date.date(),
        #     max_value=max_date.date(),
        #     value=(min_date.date(), max_date.date()),
        #     format="DD-MM-YYYY",
        # )
        # start_date, end_date = date_range

        # Apply filters
        filtered_sentiment = sentiment_df[
            (sentiment_df["Keyword"].isin(selected_keywords)) &
            (sentiment_df["Label"].isin(selected_labels)) &
            (sentiment_df["Timestamp"].dt.date >= start_date) &
            (sentiment_df["Timestamp"].dt.date <= end_date)
        ]

        # Show summary
        st.write(f"**Filtered Data Count:** {len(filtered_sentiment)}")
        st.write(f"**Date Range:** {start_date} to {end_date}")
        st.write(f"**Selected Keywords:** {', '.join(selected_keywords)}")
        st.write(f"**Selected Sentiment Labels:** {', '.join(selected_labels)}")

        # Search functionality
        search_text = st.text_input("Search Post Text")
        if search_text:
            filtered_sentiment = filtered_sentiment[
                filtered_sentiment["Original_Text"].str.contains(search_text, case=False, na=False)
            ]

        # Show tip with close option
        if 'show_tip' not in st.session_state:
            st.session_state.show_tip = True

        if st.session_state.show_tip:
            col1, col2 = st.columns([10, 1])
            with col1:
                st.info("💡 **Tip:** You can click and drag column headers to rearrange them, and click on column headers to sort the data.")
            with col2:
                if st.button("✕", key="close_sentiment_tip", help="Close tip"):
                    st.session_state.show_tip = False
                    # st.rerun()

        # Show filtered data
        st.dataframe(
            filtered_sentiment.reset_index(drop=True),
            use_container_width=True,
            height=500,
        )

    # ------------------- #
    # Trend Tab
    # ------------------- #
    with tab2:
        st.subheader("Search Trend Data Explorer")

        # Example trend_df columns: ['Timestamp', 'Keyword', 'Volume', 'Price']
        trend_keywords = sorted(trend_df["Keyword"].unique())
        trend_start, trend_end = trend_df["Date"].min(), trend_df["Date"].max()

        selected_trend_keywords = st.multiselect("Select Keywords", trend_keywords, default=trend_keywords)
        trend_start_date, trend_end_date = st.date_input("Trend Date Range", [trend_start.date(), trend_end.date()])

        # Filter trend data
        filtered_trend = trend_df[
            (trend_df["Keyword"].isin(selected_trend_keywords)) &
            (trend_df["Date"].dt.date >= trend_start_date) &
            (trend_df["Date"].dt.date <= trend_end_date)
        ]

        # Show summary
        st.write(f"**Filtered Trend Data Count:** {len(filtered_trend)}")
        st.write(f"**Trend Date Range:** {trend_start_date} to {trend_end_date}")
        st.write(f"**Selected Keywords:** {', '.join(selected_trend_keywords)}")

        # Search functionality
        search_trend_text = st.text_input("Search Trend Text")
        if search_trend_text:
            filtered_trend = filtered_trend[
                filtered_trend["Keyword"].str.contains(search_trend_text, case=False, na=False)
            ]
        # Show tip with close option
        if 'show_tip' not in st.session_state:
            st.session_state.show_tip = True
        if st.session_state.show_tip:
            col1, col2 = st.columns([10, 1])
            with col1:
                st.info("💡 **Tip:** You can click and drag column headers to rearrange them, and click on column headers to sort the data.")
            with col2:
                if st.button("✕", key="close_trend_tip", help="Close tip"):
                    st.session_state.show_tip = False
                    st.rerun()

        st.dataframe(
            filtered_trend.reset_index(drop=True),
            use_container_width=False,
            height=500,
        )

    # ------------------- #
    # Price Data Tab
    # ------------------- #
    with tab3:
        st.subheader("Generated Price Data")
        sentiment_df = load_sentiment_data()
        trends_df = load_trends_data()
        price_df = create_sample_price_data(sentiment_df, trends_df)
        combined_df = combine_all_data(price_df, sentiment_df, trends_df)

        st.write(f"Showing {len(combined_df)} records")
        print(combined_df.head())

        st.dataframe(combined_df, use_container_width=True)

        # Download option
        csv = combined_df.to_csv(index=False)
        st.download_button("Download Combined Data", csv, "combined_data.csv", "text/csv")

        # Data summary
        st.subheader("📊 Data Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", len(combined_df))

        with col2:
            st.metric(
                "Date Range",
                f"{(combined_df['Date'].max() - combined_df['Date'].min()).days} days",
            )

        with col3:
            st.metric("Avg Price", f"${combined_df['Close'].mean():.4f}")

        with col4:
            st.metric("Price Volatility", f"{combined_df['Close'].std():.4f}")
