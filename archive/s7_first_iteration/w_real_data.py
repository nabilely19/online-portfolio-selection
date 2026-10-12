import VB_FTRL 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from price import rescale
import pandas as pd
import yfinance as yf

tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA',
    'META', 'JPM', 'JNJ', 'V', 'PG', 'HD', 'DIS', 'MA', 
    'VZ', 'CMCSA', 'NFLX', 'PFE', 'PEP', 'T', 'MRK', 
    'CSCO', 'INTC', 'NKE', 'XOM', 'TMO', 'CVX', 'ABT', 
    'IBM', 'MDT', 'ACN', 'CRM', 'AVGO', 'HON', 'LLY', 
    'QCOM', 'COST', 'AMGN', 'TXN', 'AMAT', 'NOW', 
    'LMT', 'BMY', 'NEE', 'SBUX', 'DHR', 'SYK', 'CAT'
]

class parameters:
    def __init__(self, period, interval, n = None):
        self.period = period
        self.interval = interval
        self.n = n

param = parameters("2y","1d")

def getdata(parameters):
    data = yf.download(tickers, period=parameters.period, interval=parameters.interval)
    return data.ffill().bfill().dropna(axis=1)

data =  getdata(param)
columns_prices = [("Close", ticker) for ticker in tickers]

prix = data[columns_prices]
returns = (prix.pct_change()).iloc[1:]
rescaled = rescale(returns, "minmax", treshold=0.09)

param.n = 48
param.n = rescaled.shape[1]
T = prix.shape[0]

capital_algo = np.zeros(T)
capital_algo[0] = 1

capital_uniform = np.zeros(T)
capital_uniform[0] = 1

algo = VB_FTRL.VB_FTRL(param.n, mu = 1/5  , lambda_val= 1/5)
pastw = [algo.w]

# f = plt.figure(figsize=(25, 20))
# plt.matshow(returns.corr(), fignum=f.number)
# plt.xticks(range(returns.select_dtypes(['number']).shape[1]), returns.columns.droplevel(0), fontsize=7, rotation=45)
# plt.yticks(range(returns.select_dtypes(['number']).shape[1]), returns.columns.droplevel(0), fontsize=7)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16);
# plt.show()

for tau in range(T-1):
    w = algo.w
    capital_algo[tau+1] = capital_algo[tau]*np.dot(w,returns.iloc[tau] + 1)
    capital_uniform[tau+1] = capital_uniform[tau]*np.sum(returns.iloc[tau] + 1)/param.n
    algo.update(rescaled.iloc[tau])
    algo.w[algo.w < 0] = 0.1
    algo.w = algo.w / np.linalg.norm(algo.w,1)
    pastw.append(algo.w)
    # print(f"\rmarket :{capital_uniform[tau+1]:.4f}, algo : {capital_algo[tau+1]:.4f}, {capital_algo[tau+1] - capital_uniform[tau+1]:.4f}", f'{max(algo.w):.4f}', f'{min(algo.w):.4f}', end="")



####################################################################
fig, ax = plt.subplots(nrows=2, ncols=2)
plt.rcParams['figure.figsize'] = [18, 14]
x_dates = prix.index
####################################################################
ax[0][0].set_title("les actifs")
# ax[0][0].set_xlabel("date")
ax[0][0].set_ylabel("prix")
ax[0][0].plot(prix)
# ax[0][0].set_yscale("log")
ax[0][0].xaxis.set_major_locator(MaxNLocator(nbins=5))
####################################################################
ax[0][1].set_title("évolution des actifs par rapport au prix initial")
# ax[0][1].set_xlabel("date")
ax[0][1].set_ylabel("prix")
ax[0][1].plot(prix/prix.iloc[0])
ax[0][1].set_yscale("log")
ax[0][1].xaxis.set_major_locator(MaxNLocator(nbins=5))
####################################################################
ax[1][0].set_title("Poids du portefeuille de l'algorithme")
ax[1][0].plot(x_dates,pastw)
# ax[1][0].set_xlabel("date")
ax[1][0].xaxis.set_major_locator(MaxNLocator(nbins=5))
####################################################################
ax[1][1].set_title("Valeurs du portefeuille de l'algorithme et du portefeuille équiréparti")
ax[1][1].plot(x_dates,capital_algo, label = 'Algorithme')
ax[1][1].plot(x_dates,capital_uniform, label = 'Portefeuille équiréparti')
ax[1][1].set_yscale("log")
# ax[1][1].set_xlabel("date")
ax[1][1].xaxis.set_major_locator(MaxNLocator(nbins=5))
ax[1][1].legend()

plt.show()
