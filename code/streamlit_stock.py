import streamlit as st 

import pandas as pd 
import numpy as np 
from pandas_datareader import data as pdr
import yfinance as yfin
from prophet import Prophet
from prophet.plot import (
    plot_plotly,
    plot_components_plotly,
    plot_forecast_component_plotly,
    plot_seasonality_plotly
)

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from plotly.subplots import make_subplots


import talib as ta 



st.write("# Stock Analysis Dashboard by James")



ticker = st.text_input("Please input the 4 letter code for the stock you would like to investigate")
period = st.selectbox("What seasonality do you want to use in the FB Prophet model?",
                      ("Year", 
                       "1/2 year", 
                       "Quarterly", 
                       "Monthly", 
                       "Daily"))

date = st.text_input("What date do you want to start the analysis from? please do it in the form YYYY-MM-DD")


if st.button('Run Analysis on Selected Steps'):
    
    if period == 'Year':
        seasonality = 1
    if period =='1/2 year':
        seasonality = 2
    if period == 'Quarterly':
        seasonality = 4
    if period == 'Monthly':
        seasonality = 12
    if period == 'Daily':
        seasonality = 365
        
    
    yfin.pdr_override()
    
    data = pdr.get_data_yahoo(ticker,  start=date)
    d2 = data.reset_index().rename(columns = {'Date':'ds', 'Close':'y'})
    d3  = d2.rename(columns = {'ds':'Date',
                          'y':'Close'})
    
    s_p = pdr.get_data_yahoo('^GSPC',  start=date)
    
    
    def make_candlestick(df):
        fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'])],

                    )

        fig.update_layout(
            autosize=False,
            width = 900,
            height= 500,)
        
        
        fig.update_layout(
        title="Candle Stick Chart for "+ticker,
        xaxis_title="Date",
        yaxis_title="Price USD",
        font=dict(
            family="Times New Roman",
            size=13,
            color="#7f7f7f"
        )
    )
        
        st.plotly_chart(fig, theme = 'streamlit')
        
    make_candlestick(d3)


    def make_macd(df):
        macd, macd_signal, macd_hist = ta.MACD(df['Close'])
        mcd = pd.DataFrame(columns =  ['MACD', 
                                      'MACD_Signal',  'MACD_hist'],
                           )
        mcd['MACD'] = macd
        mcd['MACD_Signal'] = macd_signal
        mcd['MACD_hist']=macd_hist
        mcd['Close'] = data['Close']
        #mcd = mcd.reset_index()
        mcd = mcd.dropna()
        
        return mcd
    
    dax =  make_macd(data)
    
    def show_macd(df):
        
        fig = go.Figure()

        trace_dax_close = go.Scatter(x=dax.index, y=df['Close'], name=ticker+' Close')
        trace_macd = go.Scatter(x=dax.index, y=df['MACD'], name='MACD',mode='lines')
        trace_signal = go.Scatter(x=dax.index, y=df['MACD_Signal'], name='Signal',mode='lines')

        # Create trace for histogram
        colors = np.array(['green' if x>0 else 'red' for x in dax['MACD_hist']])
        trace_hist = go.Bar(x=df.index, y=df['MACD_hist'], name='Histogram',marker=dict(color=colors))
        # Create subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

        # Add trace to subplot
        fig.add_trace(trace_dax_close, row=1, col=1)
        fig.add_trace(trace_macd, row=2, col=1)
        fig.add_trace(trace_signal, row=2, col=1)
        fig.add_trace(trace_hist, row=2, col=1)
        
        fig.update_layout(
        yaxis_title="MACD",
        font=dict(
            family="Times New Roman",
            size=13,
            color="#7f7f7f"
        )
    )
        

        # Update layout
        fig.update_layout(title=  ticker+' Close and MACD', xaxis=dict(title='Date'), yaxis=dict(title='Price'),
                          
                          
                         font=dict(
            family="Times New Roman",
            size=13,
            color="#7f7f7f"
        ))

        fig.update_layout(
            autosize=False,
            width = 900,
            height=500,)

        # Show the plot
        #fig.show()
        st.plotly_chart(fig, theme = 'streamlit')


    show_macd(dax)

    
    def show_MA(df):  
        
        
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        
        df = df.dropna()
        
        fig =  go.Figure()




        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     showlegend=False, 
                                   ))
        #add moving average traces
        fig.add_trace(go.Scatter(x=df.index,
                                 y=df['MA5'],
                                 opacity=0.7,
                                 line=dict(color='blue', width=2),
                                 name='MA 5', 
                                ))
        fig.add_trace(go.Scatter(x=df.index,
                                 y=df['MA20'],
                                 opacity=0.7,
                                 line=dict(color='orange', width=2),
                                 name='MA 20', 

                                ))
        # hide dates with no values
        #fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
        # remove rangeslider
        fig.update_layout(xaxis_rangeslider_visible=False)
        # add chart title
        fig.update_layout(
            autosize=False,
            width = 800,
            height=600,)
    
        fig.update_layout(
        title= ticker + "5 and 20 Day Moving Average",
        xaxis_title="Date",
        yaxis_title="Price USD",
        font=dict(
            family="Times New Roman",
            size=13,
            color="#7f7f7f"
        ))
    
      
        
        st.plotly_chart(fig, theme = 'streamlit')
    
        
    show_MA(data) 
    
    
    
    def show_RSI(df):
        
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(x =df.index,
                                 y = ta.RSI(df['Close']),
                                 opacity=0.7,
                                 #line=dict(color='orange', width=2),
                                 name=ticker, 

                                ))
        fig.add_hline(y=70,  line_dash="dash", line_color="red", opacity = .7)
        fig.add_hline(y=30,  line_dash="dash", line_color="green", opacity = .7)
        fig.update_layout(
                xaxis_title="Date",
                yaxis_title="RSI",
                )

        fig.update_layout(
                              yaxis = dict(
                                tickmode='array', #change 1
                                tickvals = [20,30,40,50,60,70,80,], #change 2
                                ticktext = ['20', '30', '40', '50',
                                           '60', '70', '80'], #change 3
                                ))

        fig.add_trace(go.Scatter(x=s_p.index,
                                 y=ta.RSI(s_p['Close']),
                                 opacity=0.7,
                                 line=dict(color='orange', width=2),
                                 name='S&P 500', 

                                ))

        fig.update_layout(
            autosize=False,
            width = 900,)
        st.plotly_chart(fig, theme = 'streamlit')
    
    
    show_RSI(data)
    
    
    def show_stdev(df):
        #fig = px.line(x =df.index, y = ta.STDDEV(df['Close']), title='Sandard Dev of TICKER', )
        fig = go.Figure()


        fig.add_trace(go.Scatter(x =df.index,
                                 y = ta.STDDEV(df['Close']),
                                 opacity=0.7,
                                 #line=dict(color='orange', width=2),
                                 name=ticker, 

                                ))



        fig.update_layout(
                xaxis_title="Date",
                yaxis_title="ST Dev ",
                )
        fig.update_layout(
            autosize=False,
            width = 900,)

        st.plotly_chart(fig, theme = 'streamlit')

    show_stdev(data)
    
    
    
    def prophet(df):
        
        model = Prophet(interval_width=0.95)
        model.fit(d2)

        forecast = model.make_future_dataframe(periods=seasonality, freq='MS',                              
                                              )
        forecast = model.predict(forecast)
        fig = plot_plotly(model, forecast, trend=True, )
        fig.update_layout(
                title="FB Prophet Projections for  AAPL",
                xaxis_title="Date",
                yaxis_title="Price USD",
                font=dict(
                    family="Times New Roman",
                    size=13,
                    color="#7f7f7f"
                )
            )
        st.plotly_chart(fig, theme = 'streamlit')
        
    prophet(d2)
    
    
    
    s_d =  yfin.Ticker(ticker).info
    st.write("Trailing PEG Ratio:", s_d['trailingPegRatio'])
    st.write("Market Cap:", s_d['marketCap'])
    st.write("Enterprise to EBITDA:", s_d['enterpriseToEbitda'])
    st.write("EBITDA to margins :", s_d['ebitdaMargins'])
    st.write("Current Price: ", s_d['currentPrice'])
    st.write('Trailing PE: ', s_d['trailingPE'])
    st.write('Forward PE: ', s_d['forwardPE'])
    st.write("Peg Ratio:", s_d['pegRatio'])
    st.write("Total Debt/Revenue:", s_d['totalDebt'] / s_d['totalRevenue'])
    st.write("Debt to Equity:", s_d['debtToEquity'])
    st.write("Trailing Annual Dividend Yield :",  s_d['trailingAnnualDividendYield']) 
    st.write(" ")
    st.write("Current Price:", s_d['currentPrice'])
    st.write("Recommendation:", s_d['recommendationKey'].upper())
    st.write('Number of Analysts Opinions:', s_d['numberOfAnalystOpinions'])


st.write(" ")
st.write(" ")
st.write(" ") 
st.markdown("""
##### This is not to be taken as financial advice.
This is to be taken as data cultivation, gathering and understanding

For questions, please contact me at jamesgriffd@gmail.com or on linkedin James Griffin-Gomez



""")