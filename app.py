from flask import Flask, render_template, request, jsonify
import yfinance as yf
import plotly.graph_objs as go
import requests
from datetime import datetime
import calendar
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Your NewsAPI key
NEWS_API_KEY = '656a207c314d4439b45406ddadbb161a'

# Predefined stock tickers and their full names
STOCKS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "TSLA": "Tesla, Inc.",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.",
    "FB": "Meta Platforms, Inc.",
    "NFLX": "Netflix, Inc.",
    "NVDA": "NVIDIA Corporation",
    "BABA": "Alibaba Group Holding Limited",
    "SPCE": "Virgin Galactic Holdings, Inc."
}

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    return hist

def create_stock_chart(hist, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=f'{ticker} Stock Price Over the Last Year',
                      xaxis_title='Date',
                      yaxis_title='Price (USD)')
    graph = fig.to_html(full_html=False)
    return graph

def get_news(ticker):
    url = f'https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={NEWS_API_KEY}&language=en'
    response = requests.get(url)
    news = response.json()
    articles = news.get('articles', [])[:5]  # Get top 5 news articles
    return articles

def analyze_stock(ticker, discount_rate=0.15):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    avg_price = hist['Close'].mean()
    current_price = hist['Close'][-1]

    # Fetch full stock name if available
    stock_full_name = STOCKS.get(ticker, ticker)

    # Simple trend analysis (bullish or bearish)
    trend_direction = "Bullish" if hist['Close'][-1] > hist['Close'][0] else "Bearish"

    # Linear regression to predict future prices (next 6 months)
    hist['Days'] = np.arange(len(hist))
    X = hist[['Days']]
    y = hist['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.array([[len(hist) + i * 30] for i in range(1, 7)])  # Next 6 months
    future_prices = model.predict(future_days)

    prediction_table = {
        "Predicted Price": [round(price, 2) for price in future_prices]
    }

    if current_price > avg_price:
        valuation = 'Overvalued'
        valuation_detail = (
            f"The current price is above the average price of the last year. The stock is considered "
            f"overvalued. The market expects high growth from this company in the future. "
            f"However, be cautious as overvaluation could lead to a correction."
        )
    else:
        valuation = 'Undervalued'
        valuation_detail = (
            "The stock is currently trading below its average price over the past year, indicating it may be undervalued. "
            "This could present a buying opportunity for investors."
        )

    return {
        'current_price': round(current_price, 2),
        'average_price': round(avg_price, 2),
        'valuation': valuation,
        'valuation_detail': valuation_detail,
        'trend_direction': trend_direction,
        'stock_full_name': stock_full_name,
        'prediction_table': prediction_table
    }

def add_months(start_date, months):
    """
    Function to calculate future months based on the current date.
    :param start_date: The starting date (typically current date)
    :param months: Number of months to add
    :return: List of future month names
    """
    future_months = []
    for i in range(months):
        month = (start_date.month - 1 + i) % 12 + 1
        year = start_date.year + ((start_date.month - 1 + i) // 12)
        future_months.append(f"{calendar.month_name[month]} {year}")
    return future_months

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        hist = get_stock_data(ticker)
        chart = create_stock_chart(hist, ticker)
        news = get_news(ticker)
        analysis = analyze_stock(ticker)

        # Get current date and calculate next 6 months
        now = datetime.now()
        future_months = add_months(now, 6)

        return render_template('index.html', chart=chart, news=news, analysis=analysis, ticker=ticker, future_months=future_months)
    return render_template('index.html')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search = request.args.get('q').upper()
    results = [ticker for ticker in STOCKS.keys() if ticker.startswith(search)]
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
