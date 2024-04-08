from flask import (Flask, flash, redirect, render_template, request, url_for)
import subprocess
import io
import base64


from pypfopt import expected_returns
from pypfopt import risk_models
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

# creating demo user portfolio
# other crypto currencies or stocks
y_symbols = ['PI', 'BTC', 'BNB', 'SOL', 'OXY']

# starting date for stocks
stockStartDate = '2020-01-01'

# ending sate for stocks
today = datetime.today().strftime('%Y-%m-%d')
# print(today)

# close price of stocks
data = pdr.get_data_yahoo(y_symbols, stockStartDate, today)['Adj Close']
# print(data)

# visual representation of portfolio'
title = 'Portfolio Adj. Close Price History'

# getting stocks
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
    # yf.pdr_override()

    # plt.style.use('fivethirtyeight')

    # # creating demo user portfolio
    # # other crypto currencies or stocks
    # y_symbols = ['PI', 'BTC', 'BNB', 'SOL', 'OXY']

    # # weight to stocks
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # # starting date for stocks
    # stockStartDate = '2020-01-01'

    # # ending sate for stocks
    # today = datetime.today().strftime('%Y-%m-%d')
    # print(today)

    # # close price of stocks
    # data = pdr.get_data_yahoo(y_symbols, stockStartDate, today)['Adj Close']
    # print(data)

    # # visual representation of portfolio'
    # title = 'Portfolio Adj. Close Price History'

    # # getting stocks
    # my_stocks = data

    # creating & ploting graph
    graph_image = get_chart()

    # daily simple return
    returns = data.pct_change()
    # print(returns)

    # annual covariance matrix
    def get_annual_covariance_matrix(returns, days_per_year=252):
        return returns.cov() * days_per_year

    cov_matrix_annual = get_annual_covariance_matrix(returns)

    # cov_matrix_annual = returns.cov() * 252
    # print("cov_matrix_annual",cov_matrix_annual)

    # portfolio variance
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
    # print("portfolio_variance", portfolio_variance)

    # portfolio volatility or standard deviation
    portfolio_volatility = np.sqrt(portfolio_variance)

    # print("portfolio_volatility", portfolio_volatility)

    # annual portfolio return

    def portfolio_return(returns, weights):
        return np.sum(returns.mean() * weights) * 252

    annualPortfolioReturn = portfolio_return(returns, weights)
    # print("annual_Portfolio_Return", annualPortfolioReturn)

    # annualPortfolioReturn = np.sum(returns.mean() * weights) * 252
    # print(annualPortfolioReturn)

    # expected annual return, volatility, variance

    percent_Variance = str(round(portfolio_variance, 2) * 100) + '%'
    percent_Volatility = str(round(portfolio_volatility, 2) * 100) + '%'
    percent_Return = str(round(annualPortfolioReturn, 2) * 100) + '%'

    # print('Expected Annual Return Before Optimization: ' + percent_Return)
    # print('Annual Volatility Before Optimization: ' + percent_Volatility)
    # print('Annual Variance Before Optimization: ' + percent_Variance)

    # portfolio optimization

    # expected returns & annualised sample covariance matrix of asset returns
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)

    # optimize for max sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    # print(cleaned_weights)
    after_optimization = ef.portfolio_performance(verbose=True)

    expected_annual_return = round(after_optimization[0]*100, 2)
    annual_volatility = round(after_optimization[1]*100, 2)
    sharpe_ratio = round(after_optimization[2], 2)

    # discrete allocation of each share per stock
    # if request.method == "POST":
    #     # global total_portfolio_value
    #     total_portfolio_value=request.form.get('inputValue')

    latest_prices = get_latest_prices(data)
    da = DiscreteAllocation(
        cleaned_weights, latest_prices, total_portfolio_value=5000)
    allocation,leftover = da.greedy_portfolio()
    # leftover = da.lp_portfolio()
    # leftover = da.greedy_portfolio()
    # print('Discrete Allocation: ', allocation) # Crypto currency buying suggestion
    leftoover = round(leftover, 2)  # leftover balance


    # app.logger.info('Processing default request')

    return render_template('base.html', get_chart=graph_image, percent_Variance_be4_opt=percent_Variance,
                           percent_Volatility_be4_opt=percent_Volatility, percent_Return_be4_opt=percent_Return,
                           expected_annual_return=expected_annual_return, annual_volatility=annual_volatility, sharpe_ratio=sharpe_ratio, allocation_output=allocation, leftover_bal=leftoover,
                           today=today
                           )


if __name__ == "__main__":
    app.run(debug=True)
