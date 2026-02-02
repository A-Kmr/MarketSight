# Interactive-Stock-Market-Forecasting
# 📈 MarketSight: Stock Forecasting & Analytics Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Status](https://img.shields.io/badge/Status-Prototype-green)

**MarketSight** is an interactive web application designed to democratize financial analysis. It enables users to visualize real-time market trends, conduct technical analysis, and generate short-term price forecasts using Machine Learning.

Built with **Streamlit** and **Scikit-Learn**, this tool bridges the gap between raw financial data and actionable insights.

---

## 🚀 Key Features

### 1. Real-Time Data Ingestion
* **Live API Integration:** Fetches up-to-the-minute stock data using the **Yahoo Finance API (`yfinance`)**.
* **Dynamic Ticker Search:** Supports analysis for major global indices and equities (e.g., TSLA, AAPL, BABA).

### 2. Advanced Exploratory Data Analysis (EDA)
* **Technical Indicators:** Automatically calculates and visualizes **50-Day and 200-Day Moving Averages** to identify "Golden Cross" or "Death Cross" patterns.
* **Statistical Deep Dive:** Generates volatility distributions, correlation heatmaps, and price range analysis.
* **Interactive Visuals:** Uses **Plotly** and **Seaborn** for high-interactivity charts that allow zooming and panning.

### 3. Machine Learning Forecasting
* **Algorithm:** Utilizes a **Random Forest Regressor** ensemble model for robust prediction in volatile markets.
* **Feature Engineering:** Enhances raw price data with calculated technical features like **MACD** (Moving Average Convergence Divergence) and **EMA** (Exponential Moving Average).
* **Sliding Window Approach:** Implements a time-step based sequence generation (Look-back window) to transform time-series data into a supervised learning problem.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Web Framework)
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest, MinMax Scaling)
* **Visualization:** Plotly (Interactive), Matplotlib, Seaborn
* **Financial Data:** yfinance API
* **Statistical Analysis:** Statsmodels (Seasonal Decompose)

---

## ⚙️ Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/A-Kmr/market-sight.git](https://github.com/A-Kmr/market-sight.git)
    cd market-sight
    ```

2.  **Install Dependencies**
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    Launch the dashboard in your local browser:
    ```bash
    streamlit run app.py
    ```

---

## 🧠 Model Methodology

The forecasting engine follows a strict data pipeline to ensure reliability:

1.  **Data Cleaning:** Handling missing values and normalizing price ranges using `MinMaxScaler` (0-1 scale).
2.  **Feature Generation:** Creating lag features and technical indicators (MACD, Signal Line) to feed the model context beyond just "Price."
3.  **Sequence Creation:** The data is transformed into a sliding window structure (e.g., using the past 60 days to predict the next day).
4.  **Training:** A `RandomForestRegressor` (n=100) is trained on 86% of the historical data, with the remaining 14% used for out-of-sample validation.

---

## 🔮 Future Roadmap

* [ ] **Deep Learning Integration:** Implementing LSTM (Long Short-Term Memory) networks for improved long-term dependency capture.
* [ ] **Sentiment Analysis:** Integrating news API data to correlate price movements with market sentiment.
* [ ] **Portfolio Optimization:** Adding Markowitz Efficient Frontier analysis for asset allocation.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
*Created by Anirudh Kumar Pentapati | [LinkedIn](https://www.linkedin.com/in/anirudhkumar98)*

---

*Created by Anirudh Kumar Pentapati | [LinkedIn](https://www.linkedin.com/in/anirudhkumar98)*
