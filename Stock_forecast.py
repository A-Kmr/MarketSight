import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from itertools import cycle
import matplotlib
pd.options.plotting.backend = "matplotlib"

# Side panel links
PAGES = {
    "Home": "Home",
    "EDA": "EDA",
    "Data Preparation": "Data Preparation",
    "Modeling": "Modeling",
    "Forecasting": "Forecasting"
}

# List of sample stock tickers
SAMPLE_TICKERS = [
    "BABA", "PFE", "DIS", "BA", "TSLA", "WMT", "NFLX", "PEP", "IBM", "SPOT"]

# Function to load stock data from Yahoo Finance and get dynamic start and end dates
def load_stock_data(ticker):
    # Download stock data
    data = yf.download(ticker)
    data.columns = data.columns.droplevel(1)
    
    data = data.sort_index(ascending=False)   
    
    # Get the start and end dates dynamically from the data
    start_date = data.index.min().strftime('%Y-%m-%d')
    end_date = data.index.max().strftime('%Y-%m-%d')
    
    # Fetch the stock info (name, sector, etc.)
    stock_info = yf.Ticker(ticker).info
    stock_name = stock_info.get('longName', 'Stock name not available')  # Fetch stock name
    
    return data, start_date, end_date, stock_name

# Home Page
def home_page(ticker):
    st.title("Stock Price Forecasting")
    st.write("""Welcome to the **Stock Price Forecasting App**.""")

    st.write(f"Selected Ticker: {ticker}")

    # Load stock data based on user's selection
    data, start_date, end_date, stock_name = load_stock_data(ticker)
    st.write(f"Displaying data for {stock_name} from {start_date} to {end_date}")
    
    st.write(data)

    return ticker, data, start_date, end_date

# EDA Page
def eda_page(data):
    st.title("Statistical Insights")
    
    # UNIVARIATE ANALYSIS
    st.header("Univariate Analysis")

    # Calculate 50-day and 200-day moving averages
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['200_MA'] = data['Close'].rolling(window=200).mean()

    # Display the stock data and moving averages
    st.write("Stock data with moving averages:")
    st.write(data[['Close', '50_MA', '200_MA']].tail())

    # Plot the Closing Price and Moving Averages
    st.write("Closing Price and Moving Averages:")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.plot(data.index, data['50_MA'], label='50-Day Moving Average')
    ax.plot(data.index, data['200_MA'], label='200-Day Moving Average')
    ax.set_title(f'{data.index[0]} to {data.index[-1]} Closing Price and Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)
    

    fig = plt.figure(figsize=(10, 6))
    sns.histplot(data['Close'], kde=True)
    st.pyplot(fig)

    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(x=data['Close'])
    st.pyplot(fig)
    
    data['open-high'] = data['Open']-data['High']
    data['open-low'] = data['Open'] - data['Low']
    data['close-high'] = data['Close']-data['High']
    data['close-low'] = data['Close'] - data['Low']
    data['high-low'] = data['High'] - data['Low']
    data['open-close'] = data['Open'] - data['Close']
    data2 = data.copy()
    data2 = data2.drop(['Open','High','Low','Close', 'Adj Close','50_MA', '200_MA'],axis=1)
    
    
    # Bivariate - Correlation Heatmap
    st.subheader("Bivariate Analysis - Correlation Heatmap")
    correlation_matrix = data2.corr()
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="Blues")
    st.pyplot(fig)
    
     # Bivariate - Distribution and Box Plot
    st.subheader("Bivariate Analysis")
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x = "open-high", y="Volume")
    st.pyplot(fig)
    

# Data Preparation Function (handles data processing)
def data_preparation(data):
    data = data.sort_index(ascending=True)
    
    # Adding new features
    data['Price_Range'] = data['High'] - data['Low']
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['200_MA'] = data['Close'].rolling(window=200).mean()
    
    # Technical Indicators (MACD)
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    data = data.dropna()
    
    return data

# Data Preparation Page
def data_preparation_page(data):
    st.title("Data Preparation")
    
    st.subheader("Raw Data")
    st.write(data)
    
    data = data_preparation(data)
    data = data.sort_index(ascending=True)
    st.write("Data after feature engineering:")
    st.write(data)
    
    return data  # Return cleaned data after all the actions


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)





# Modeling Page
def modeling_page(data):
    st.title("Modeling")
    data = data_preparation(data)
    
    st.write("Daily Close Prices of the stock")
    close_df=data['Close']
    close_df=close_df.reset_index()
    st.write(close_df)
    
    close_stock = close_df.copy()
    del close_df['Date']
    scaler=MinMaxScaler(feature_range=(0,1))
    closedf=scaler.fit_transform(np.array(close_df).reshape(-1,1))
    
    training_size=int(len(closedf)*0.86)
    test_size=len(closedf)-training_size
    train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
    
    
    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 13
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    st.write("Training The Model")    
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
    regressor.fit(X_train, y_train)
    st.write("Model Training Complete")
    
    
    train_predict=regressor.predict(X_train)
    test_predict=regressor.predict(X_test)

    train_predict = train_predict.reshape(-1,1)
    test_predict = test_predict.reshape(-1,1)
    
    
    # Transform back to original form

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))
    
    st.write("Original Close Price", original_ytest)
    st.write("Predicted Close Price", test_predict)
    forecast_index = close_stock['Date'][-len(test_predict):].reset_index(drop=True)
    st.write(forecast_index)
    
    # Creating the DataFrame
    forecast_data = pd.DataFrame({
        'Date':forecast_index,
        'Original Close Price': original_ytest.tolist(),
        'Predicted Close Price': test_predict.tolist()  
    })
    st.write(forecast_data)
    


    
    # Model Evaluation
    st.subheader("Model Evaluation ")
    mape = np.mean(np.abs((original_ytest - test_predict) / original_ytest)) * 100
    st.write(f"Mean Absolute Percentage Error(MAPE): {mape:.2f}%")
    
#     # Actual vs Predicted Plot
#     fig = plt.figure(figsize=(10, 6))
#     plt.plot(original_ytest.index, original_ytest, label="Actual")
#     plt.plot(original_ytest.index, test_predict, label="Predicted")
#     plt.legend()
#     st.pyplot(fig)
    
    
    # shift train predictions for plotting

    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    print("Train predicted data: ", trainPredictPlot.shape)

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
    print("Test predicted data: ", testPredictPlot.shape)

    names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


    plotdf = pd.DataFrame({'Date': close_stock['Date'],
                           'original_close': close_stock['Close'],
                          'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                          'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

    fig = px.line(plotdf,x=plotdf['Date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                              plotdf['test_predicted_close']],
                  labels={'value':'Stock price','Date': 'Date'})
    fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig)
    
    

# Forecasting Page
def forecasting_page(data):
    st.title("Forecasting")
    data = data_preparation(data)
    
    # Buttons for Short Term and Long Term forecasting
    forecast_type = st.radio("Select Forecast Type", ("Short Term (Upto a month)", "Long Term (1 month - 24 months)"))
    
    if forecast_type == "Short Term (Upto a month)":
        # Short Term Forecast: Number Picker for 1-31 days
        days_to_forecast = st.number_input("Select number of days (1-31)", min_value=1, max_value=31, step=1)
        
        if days_to_forecast:
            st.write(f"Forecasting for the next {days_to_forecast} days.")
            forecast(data, days_to_forecast)
    
    elif forecast_type == "Long Term (1 month - 24 months)":
        # Long Term Forecast: Number Picker for 1-24 months
        months_to_forecast = st.number_input("Select number of months (1-24)", min_value=1, max_value=24, step=1)
        days_to_forecast = months_to_forecast*30
        if months_to_forecast:
            st.write(f"Forecasting for the next {months_to_forecast} months.")
            forecast(data, days_to_forecast)

# Short Term Forecasting Function
def forecast(data, days_to_forecast):
    data = data_preparation(data)
    
    forecast_index = pd.date_range(start=data.index[-1], periods=days_to_forecast+1, freq='D')[1:]
    future_predictions = []
    
    close_df=data['Close']
    close_df=close_df.reset_index()
    
    close_stock = close_df.copy()
    del close_df['Date']
    scaler=MinMaxScaler(feature_range=(0,1))
    closedf=scaler.fit_transform(np.array(close_df).reshape(-1,1))
    
    training_size=int(len(closedf)*0.86)
    test_size=len(closedf)-training_size
    train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]

    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
    regressor.fit(X_train, y_train)
    
    
    train_predict=regressor.predict(X_train)
    test_predict=regressor.predict(X_test)

    train_predict = train_predict.reshape(-1,1)
    test_predict = test_predict.reshape(-1,1)

    
    
    # Transform back to original form

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 


    
    # Model Evaluation
    mape = np.mean(np.abs((original_ytest - test_predict) / original_ytest)) * 100
    
    
    # shift train predictions for plotting

    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict

    names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


    plotdf = pd.DataFrame({'Date': close_stock['Date'],
                           'original_close': close_stock['Close'],
                          'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                          'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

    
    
    
    
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()


    lst_output=[]
    n_steps=time_step
    i=0
    while(i<days_to_forecast):

        if(len(temp_input)>time_step):

            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)

            yhat = regressor.predict(x_input)
            temp_input.extend(yhat.tolist())
            temp_input=temp_input[1:]

            lst_output.extend(yhat.tolist())
            i=i+1

        else:
            yhat = regressor.predict(x_input)

            temp_input.extend(yhat.tolist())
            lst_output.extend(yhat.tolist())

            i=i+1


    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+days_to_forecast+1)

    temp_mat = np.empty((len(last_days)+days_to_forecast+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
    
    forecast_data = pd.DataFrame(next_predicted_days_value[time_step+1:], index=forecast_index, columns=['Forecasted_Close'])
    st.write(forecast_data)
    
    names = cycle(['Past close price','Predicted close price'])

    new_pred_plot = pd.DataFrame({
        'last_original_days_value':last_original_days_value,
        'next_predicted_days_value':next_predicted_days_value
    })

    fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                          new_pred_plot['next_predicted_days_value']],
                  labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text='Compare Past vs Forecasted',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig)
    
    rfdf=closedf.tolist()
    rfdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    rfdf=scaler.inverse_transform(rfdf).reshape(1,-1).tolist()[0]

    names = cycle(['Close price'])

    fig = px.line(rfdf,labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig)


# Main function to control the page navigation
def main():
    st.sidebar.title("Stock Price Forecasting")
    ticker = st.sidebar.selectbox("Select a stock ticker", SAMPLE_TICKERS)
    page = st.sidebar.radio("Select a page", list(PAGES.keys()))
    
    data, start_date, end_date, stock_name = load_stock_data(ticker)
    
    if page == "Home":
        home_page(ticker)
    elif page == "EDA":
        eda_page(data)
    elif page == "Data Preparation":
        data_preparation_page(data)
    elif page == "Modeling":
        modeling_page(data)
    elif page == "Forecasting":
        forecasting_page(data)

if __name__ == "__main__":
    main()
