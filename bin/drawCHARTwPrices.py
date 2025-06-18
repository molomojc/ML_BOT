from flask import Flask, render_template
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def candlestick_chart():
    # Fetch BTC-USD data (1-minute intervals for the last day)
    data = yf.download("BTC-USD", period="1d", interval="1h")
     
    # Flatten multi-level column names
    data.columns = [col[0].lower() for col in data.columns]
    
    # Create candlestick figure
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    
    # Update layout
    fig.update_layout(
        title='BTC/USD Candlestick Chart (1m)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    # Convert to HTML
    chart_html = fig.to_html(full_html=False)
    
    return render_template('chart.html', 
                        chart=chart_html,
                        last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == '__main__':
    app.run(debug=False)