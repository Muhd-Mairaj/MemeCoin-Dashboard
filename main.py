import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
import json
import ccxt
from datetime import datetime, timedelta
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
    
    # Load data
    sentiment_df = load_sentiment_data()
    trends_df = load_trends_data()
    
    # Create sample price data
    price_df = create_sample_price_data(sentiment_df, trends_df)
    combined_df = combine_all_data(price_df, sentiment_df, trends_df)
    
    # Calculate correlations
    correlation_data = combined_df[['Close', 'Sentiment_Score', 'Search_Score']].corr()
    
    # Display correlation heatmap
    st.subheader("📊 Correlation Matrix")
    fig_corr = px.imshow(correlation_data, 
                        text_auto=True, 
                        aspect="auto",
                        title="Correlation between Price, Sentiment, and Search Trends",
                        color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Price vs Sentiment")
        fig_scatter1 = px.scatter(combined_df, 
                                 x='Sentiment_Score', 
                                 y='Close',
                                 trendline="ols",
                                 title="Price vs Sentiment Score")
        st.plotly_chart(fig_scatter1, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Price vs Search Trends")
        fig_scatter2 = px.scatter(combined_df, 
                                 x='Search_Score', 
                                 y='Close',
                                 trendline="ols",
                                 title="Price vs Search Score")
        st.plotly_chart(fig_scatter2, use_container_width=True)
    
    # Time series comparison
    st.subheader("📈 Time Series Comparison")
    
    # Normalize data for comparison
    normalized_df = combined_df.copy()
    for col in ['Close', 'Sentiment_Score', 'Search_Score']:
        normalized_df[f'{col}_norm'] = (normalized_df[col] - normalized_df[col].min()) / (normalized_df[col].max() - normalized_df[col].min())
    
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=normalized_df['Date'], 
                                 y=normalized_df['Close_norm'], 
                                 mode='lines', 
                                 name='Price (Normalized)',
                                 line=dict(color='blue')))
    fig_time.add_trace(go.Scatter(x=normalized_df['Date'], 
                                 y=normalized_df['Sentiment_Score_norm'], 
                                 mode='lines', 
                                 name='Sentiment (Normalized)',
                                 line=dict(color='green')))
    fig_time.add_trace(go.Scatter(x=normalized_df['Date'], 
                                 y=normalized_df['Search_Score_norm'], 
                                 mode='lines', 
                                 name='Search Trends (Normalized)',
                                 line=dict(color='red')))
    
    fig_time.update_layout(title="Normalized Time Series Comparison",
                          xaxis_title="Date",
                          yaxis_title="Normalized Value")
    st.plotly_chart(fig_time, use_container_width=True)

elif page == "4. Price Prediction":
    st.title("🔮 Price Prediction with ML Models")
    st.markdown("Advanced price forecasting using SVR, LSTM, and Random Forest models.")
    
    # Model selection
    col1, col2, col3 = st.columns(3)
    with col1:
        model_type = st.selectbox("Select Model", ["ensemble", "lstm", "svr", "random_forest"])
    with col2:
        days_ahead = st.number_input("Days to Predict", min_value=1, max_value=30, value=7)
    with col3:
        lookback_period = st.number_input("Lookback Period", min_value=10, max_value=100, value=60)
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'is_trained' not in st.session_state:
        st.session_state.is_trained = False
    
    # Training section
    st.subheader("🏋️ Model Training")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train Models", help="Train the prediction models with current data"):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    # Load and prepare data
                    sentiment_df = load_sentiment_data()
                    trends_df = load_trends_data()
                    price_df = create_sample_price_data(sentiment_df, trends_df)
                    combined_df = combine_all_data(price_df, sentiment_df, trends_df)
                    # Initialize and train predictor
                    st.session_state.predictor = StockPricePredictor(
                        model_type=model_type,
                        lookback_period=lookback_period
                    )
                    
                    evaluation_metrics = st.session_state.predictor.train(df=combined_df, test_size=0.2)
                    st.session_state.is_trained = True
                    
                    # Save models
                    st.session_state.predictor.save_model("trained_models.pkl")
                    
                    st.success("Models trained successfully!")
                    
                    # Display evaluation metrics
                    st.subheader("📊 Model Performance")
                    metrics = evaluation_metrics
                    print(evaluation_metrics)
                    # st.write(f"**{model_name.upper()} Model:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("RMSE", f"{metrics['rmse']:.4f}")
                    with col2:
                        st.metric("MAE", f"{metrics['mae']:.4f}")
                    with col3:
                        st.metric("R²", f"{metrics['r2']:.4f}")
                    with col4:
                        st.metric("MAE", f"{metrics['mae']:.2f}%")
                    st.write("---")
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    
    with col2:
        if st.button("Load Saved Models", help="Load previously trained models"):
            try:
                st.session_state.predictor = StockPricePredictor()
                st.session_state.predictor.load_models("trained_models.pkl")
                st.session_state.is_trained = True
                st.success("Models loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load models: {str(e)}")
    
    # Prediction section
    if st.session_state.is_trained and st.session_state.predictor:
        st.subheader("🔮 Make Predictions")
        
        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                try:
                    # Load latest data
                    sentiment_df = load_sentiment_data()
                    trends_df = load_trends_data()
                    price_df = create_sample_price_data(sentiment_df, trends_df)
                    combined_df = combine_all_data(price_df, sentiment_df, trends_df)
                    
                    # Make predictions
                    predictions = st.session_state.predictor.predict(combined_df, days_ahead=days_ahead)
                    
                    # Handle different prediction return formats
                    if isinstance(predictions, dict):
                        # Dictionary format: {model_name: predictions_array}
                        pred_dict = predictions
                    elif isinstance(predictions, (np.ndarray, list)):
                        # Array format: single prediction array
                        pred_dict = {model_type: predictions}
                    else:
                        st.error("Unexpected prediction format returned")
                        st.stop()
                    
                    # Display predictions
                    st.subheader("📈 Prediction Results")
                    
                    # Current price
                    current_price = combined_df['Close'].iloc[-1]
                    st.metric("Current Price", f"${current_price:.4f}")
                    
                    # Prediction cards
                    cols = st.columns(len(pred_dict))
                    for idx, (model_name, pred_values) in enumerate(pred_dict.items()):
                        with cols[idx]:
                            # Handle different prediction formats
                            if isinstance(pred_values, (list, np.ndarray)) and len(pred_values) > 0:
                                next_price = float(pred_values[0])
                                avg_price = float(np.mean(pred_values))
                            elif isinstance(pred_values, (int, float)):
                                next_price = float(pred_values)
                                avg_price = next_price
                            else:
                                st.error(f"Invalid prediction format for {model_name}")
                                continue
                            
                            price_change = ((next_price - current_price) / current_price) * 100
                            
                            st.metric(
                                label=f"{model_name.upper()} Next Day",
                                value=f"${next_price:.4f}",
                                delta=f"{price_change:+.2f}%"
                            )
                            
                            # Show average if multiple predictions
                            if isinstance(pred_values, (list, np.ndarray)) and len(pred_values) > 1:
                                avg_change = ((avg_price - current_price) / current_price) * 100
                                st.caption(f"Avg: ${avg_price:.4f} ({avg_change:+.2f}%)")
                    
                    # Visualization
                    st.subheader("📊 Price Prediction Visualization")
                    
                    # Create future dates
                    last_date = combined_df['Date'].max()
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')
                    
                    # Plot historical and predicted prices
                    fig = go.Figure()
                    
                    # Historical prices
                    fig.add_trace(go.Scatter(
                        x=combined_df['Date'].tail(30),
                        y=combined_df['Close'].tail(30),
                        mode='lines+markers',
                        name='Historical Price',
                        line=dict(color='blue')
                    ))
                    
                    # Predictions
                    colors = ['red', 'green', 'orange', 'purple']
                    for idx, (model_name, pred_values) in enumerate(pred_dict.items()):
                        # Handle different prediction formats
                        if isinstance(pred_values, (list, np.ndarray)) and len(pred_values) > 0:
                            y_values = [float(p) for p in pred_values[:days_ahead]]
                            # Pad with last value if needed
                            while len(y_values) < days_ahead:
                                y_values.append(y_values[-1])
                        elif isinstance(pred_values, (int, float)):
                            y_values = [float(pred_values)] * days_ahead
                        else:
                            continue
                        
                        fig.add_trace(go.Scatter(
                            x=future_dates[:len(y_values)],
                            y=y_values,
                            mode='lines+markers',
                            name=f'{model_name.upper()} Prediction',
                            line=dict(color=colors[idx % len(colors)], dash='dash')
                        ))
                    
                    fig.update_layout(
                        title='Price Prediction Forecast',
                        xaxis_title='Date',
                        yaxis_title='Price ($)',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance (for Random Forest)
                    if 'random_forest' in st.session_state.predictor.models:
                        st.subheader("🎯 Feature Importance")
                        rf_model = st.session_state.predictor.models['random_forest']
                        feature_importance = rf_model.feature_importances_
                        
                        importance_df = pd.DataFrame({
                            'Feature': st.session_state.predictor.feature_columns,
                            'Importance': feature_importance
                        }).sort_values('Importance', ascending=False).head(10)
                        
                        fig_importance = px.bar(importance_df, 
                                              x='Importance', 
                                              y='Feature',
                                              orientation='h',
                                              title='Top 10 Most Important Features')
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
    else:
        st.info("Please train or load models first to make predictions.")

elif page == "5. Memecoins vs Traditional Coins":
    st.title("⚖️ Memecoins vs Traditional Coins")
    st.markdown("Comparing price behavior and volatility between memecoins and established cryptocurrencies.")
    
    # Load data
    sentiment_df = load_sentiment_data()
    trends_df = load_trends_data()
    memecoin_price_df = create_sample_price_data(sentiment_df, trends_df)
    
    # Create traditional coin data (Bitcoin simulation)
    btc_dates = pd.date_range(start=memecoin_price_df['Date'].min(), 
                             end=memecoin_price_df['Date'].max(), freq='D')
    np.random.seed(123)
    btc_prices = []
    current_btc = 45000
    
    for _ in btc_dates:
        change = np.random.normal(0, 0.03)  # Lower volatility for BTC
        current_btc = max(1000, current_btc * (1 + change))
        btc_prices.append(current_btc)
    
    btc_df = pd.DataFrame({
        'Date': btc_dates,
        'Close': btc_prices,
        'Type': 'Traditional (BTC)'
    })
    
    memecoin_compare_df = memecoin_price_df[['Date', 'Close']].copy()
    memecoin_compare_df['Type'] = 'Memecoin (DOGE)'
    
    # Normalize prices for comparison
    memecoin_compare_df['Normalized_Price'] = (memecoin_compare_df['Close'] / memecoin_compare_df['Close'].iloc[0]) * 100
    btc_df['Normalized_Price'] = (btc_df['Close'] / btc_df['Close'].iloc[0]) * 100
    
    compare_df = pd.concat([memecoin_compare_df, btc_df])
    
    # Price comparison
    st.subheader("📈 Normalized Price Comparison")
    fig_compare = px.line(compare_df, 
                         x='Date', 
                         y='Normalized_Price',
                         color='Type',
                         title='Memecoin vs Traditional Crypto Performance (Base 100)')
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # Volatility analysis
    st.subheader("📊 Volatility Analysis")
    
    memecoin_volatility = memecoin_compare_df['Close'].pct_change().std() * 100
    btc_volatility = btc_df['Close'].pct_change().std() * 100
    
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
    
    memecoin_returns = memecoin_compare_df['Close'].pct_change().dropna() * 100
    btc_returns = btc_df['Close'].pct_change().dropna() * 100
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=memecoin_returns, name='Memecoin Returns', opacity=0.7, nbinsx=30))
    fig_dist.add_trace(go.Histogram(x=btc_returns, name='Bitcoin Returns', opacity=0.7, nbinsx=30))
    fig_dist.update_layout(title='Distribution of Daily Returns (%)', 
                          xaxis_title='Daily Return (%)',
                          yaxis_title='Frequency',
                          barmode='overlay')
    st.plotly_chart(fig_dist, use_container_width=True)

elif page == "6. Raw Data Explorer":
    st.title("🧾 Raw Data Viewer")
    st.markdown("View and filter all raw datasets used in this project.")
    
    # Data selection tabs
    tab1, tab2, tab3 = st.tabs(["💬 Sentiment Data", "🔍 Search Trends", "💰 Price Data"])
    
    with tab1:
        st.subheader("Sentiment Analysis Data")
        sentiment_df = load_sentiment_data()
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            keywords = st.multiselect("Filter by Keyword", 
                                    sentiment_df['Keyword'].unique(), 
                                    default=sentiment_df['Keyword'].unique()[:2])
        with col2:
            sentiment_types = st.multiselect("Filter by Sentiment", 
                                           sentiment_df['Label'].unique(),
                                           default=sentiment_df['Label'].unique())
        
        # Date range
        date_range = st.date_input("Select Date Range",
                                  value=[sentiment_df['Timestamp'].min().date(),
                                        sentiment_df['Timestamp'].max().date()],
                                  min_value=sentiment_df['Timestamp'].min().date(),
                                  max_value=sentiment_df['Timestamp'].max().date())
        
        # Apply filters
        filtered_sentiment = sentiment_df[
            (sentiment_df['Keyword'].isin(keywords)) &
            (sentiment_df['Label'].isin(sentiment_types)) &
            (sentiment_df['Timestamp'].dt.date >= date_range[0]) &
            (sentiment_df['Timestamp'].dt.date <= date_range[1])
        ]
        
        st.write(f"Showing {len(filtered_sentiment)} records")
        st.dataframe(filtered_sentiment, use_container_width=True)
        
        # Download option
        csv = filtered_sentiment.to_csv(index=False)
        st.download_button("Download Filtered Data", csv, "sentiment_data.csv", "text/csv")
    
    with tab2:
        st.subheader("Google Trends Search Data")
        trends_df = load_trends_data()
        
        # Filters
        keywords = st.multiselect("Filter by Keyword", 
                                trends_df['Keyword'].unique(), 
                                default=trends_df['Keyword'].unique())
        
        # Apply filters
        filtered_trends = trends_df[trends_df['Keyword'].isin(keywords)]
        
        st.write(f"Showing {len(filtered_trends)} records")
        st.dataframe(filtered_trends, use_container_width=True)
        
        # Download option
        csv = filtered_trends.to_csv(index=False)
        st.download_button("Download Filtered Data", csv, "trends_data.csv", "text/csv")
    
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
            st.metric("Date Range", f"{(combined_df['Date'].max() - combined_df['Date'].min()).days} days")
        with col3:
            st.metric("Avg Price", f"${combined_df['Close'].mean():.4f}")
        with col4:
            st.metric("Price Volatility", f"{combined_df['Close'].std():.4f}")

