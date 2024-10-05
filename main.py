from flask import Flask, render_template
import subprocess
import io
import base64
from pypfopt import expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.efficient_frontier import EfficientFrontier
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf

app = Flask(__name__)
yf.pdr_override()
plt.style.use('ggplot')

y_symbols = ['PI', 'BTC', 'BNB', 'SOL', 'OXY']
stockStartDate = '2020-01-01'
today = datetime.today().strftime('%Y-%m-%d')
data = pdr.get_data_yahoo(y_symbols, stockStartDate, today)['Adj Close']
title = 'Portfolio Adj. Close Price History'
my_stocks = data

def get_chart():
    img = io.BytesIO()
    fig = plt.figure(figsize=(12, 6))
    for c in my_stocks.columns.values:
        plt.plot(my_stocks[c], label=c)
    plt.title(title)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Adj Price USD ($)', fontsize=18)
    plt.legend(my_stocks.columns.values, loc='upper left')
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route("/", methods=['GET'])
def index():
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    graph_image = get_chart()
    returns = data.pct_change()

    def get_annual_covariance_matrix(returns, days_per_year=252):
        return returns.cov() * days_per_year

    cov_matrix_annual = get_annual_covariance_matrix(returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)

    def portfolio_return(returns, weights):
        return np.sum(returns.mean() * weights) * 252

    annualPortfolioReturn = portfolio_return(returns, weights)

    percent_Variance = str(round(portfolio_variance, 2) * 100) + '%'
    percent_Volatility = str(round(portfolio_volatility, 2) * 100) + '%'
    percent_Return = str(round(annualPortfolioReturn, 2) * 100) + '%'

    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)

    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    after_optimization = ef.portfolio_performance(verbose=True)

    expected_annual_return = round(after_optimization[0]*100, 2)
    annual_volatility = round(after_optimization[1]*100, 2)
    sharpe_ratio = round(after_optimization[2], 2)

    latest_prices = get_latest_prices(data)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=5000)
    allocation, leftover = da.greedy_portfolio()
    leftover = round(leftover, 2)

    return render_template('base.html', get_chart=graph_image, percent_Variance_be4_opt=percent_Variance,
                           percent_Volatility_be4_opt=percent_Volatility, percent_Return_be4_opt=percent_Return,
                           expected_annual_return=expected_annual_return, annual_volatility=annual_volatility, 
                           sharpe_ratio=sharpe_ratio, allocation_output=allocation, leftover_bal=leftover, today=today)

if __name__ == "__main__":
    app.run(debug=True)
