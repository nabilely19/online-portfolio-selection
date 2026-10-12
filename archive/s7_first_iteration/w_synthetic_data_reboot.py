import numpy as np
import matplotlib.pyplot as plt
from VB_FTRL import VB_FTRL
from price import returns, rescale

d = 5  # Number of assets
T = 800  # Number of time rounds
S = 10  # Number of Newton steps


asset_returns = returns(d,T)
asset_prices = np.multiply.accumulate(asset_returns,0)

asset_returns_rescaled = rescale(asset_returns - 1,method="minmax")

capital_algo = np.zeros(T)
capital_algo[0] = 1

capital_uniform = np.zeros(T)
capital_uniform[0] = 1

past_w = np.zeros((T,d))
past_w[0] = np.ones(d) / d 

past_ww = np.zeros((T,d)) # Restarted portfolio
past_ww[0] = np.ones(d) / d 
wwmean = np.zeros(T) # Restarted portfolio mean
wwstd = np.zeros(T) # Restarted portfolio standard deviation
exitcount = 0
maxexit = 5
last_restarted = 0
exits = []
capital_algo_r = np.zeros(T)
capital_algo_r[0] = 1
wwmean[0] = 1
wwstd[0] = 0
alpha = 2

past_www = np.zeros((T,d)) # Restarted portfolio
past_www[0] = np.ones(d) / d
watchdog = 5
flag = 0.05 * d
capital_algo_www = np.zeros(T)
capital_algo_www[0] = 1
castd = np.zeros(T)

algow = VB_FTRL(d, mu = 1/2, lambda_val= 1/2)
algoww = VB_FTRL(d, mu = 1/2, lambda_val= 1/2)
algowww = VB_FTRL(d, mu = 1/2, lambda_val= 1/2)


for tau in range(1,T):
    algow.update(asset_returns_rescaled[tau-1])
    algow.w[algow.w < 0] = 0.1
    algow.w = algow.w / np.linalg.norm(algow.w,1)
    past_w[tau] = algow.w

    algoww.update(asset_returns_rescaled[tau-1])
    algoww.w[algoww.w < 0] = 0.1
    algoww.w = algoww.w / np.linalg.norm(algoww.w,1)
    past_ww[tau] = algoww.w

    algowww.update(asset_returns_rescaled[tau-1])
    algowww.w[algowww.w < 0] = 0.1
    algowww.w = algowww.w / np.linalg.norm(algowww.w,1)
    past_www[tau] = algowww.w

    capital_algo[tau] = capital_algo[tau-1]*np.sum(algow.w*asset_returns[tau])
    capital_uniform[tau] = capital_uniform[tau-1]*np.sum(asset_returns[tau]/d)
    capital_algo_r[tau] = capital_algo_r[tau-1]*np.sum(algoww*asset_returns[tau])
    capital_algo_www[tau] = capital_algo_www[tau-1]*np.sum(algowww*asset_returns[tau])

    www_watch = past_www[max(0, tau-watchdog):tau+1]
    for i in range(www_watch.shape[1]):
        castd[tau] += np.std(www_watch[:,i])

    if castd[tau] > flag:
        algowww = VB_FTRL(d, mu = 1/2, lambda_val= 1/2)


    wwmean[tau] = np.mean(capital_algo_r[last_restarted:tau])
    wwstd[tau] = np.std(capital_algo_r[last_restarted:tau])
    
    if np.abs(wwmean[tau] - capital_algo_r[tau]) > alpha*wwstd[tau]:
        exitcount += 1
        if exitcount > maxexit:
            exitcount = 0
            wwmean[tau] = capital_algo_r[tau]
            wwstd[tau] = wwstd[tau-1]
            last_restarted = tau
            algoww = VB_FTRL(d, mu = 1/2, lambda_val= 1/2)

    exits.append(exitcount)




'''
fig, axs = plt.subplots(nrows=5, ncols=1)
axs[0].plot(asset_prices)
axs[1].plot(w)
axs[2].plot(ww)
axs[3].plot(capital_algo, label='Capital with FTRL-qN')
axs[3].plot(capital_uniform, label='Capital with Uniform Portfolio')
axs[3].plot(capital_algo_r, label='Capital with FTRL-qN + Restart')
axs[3].legend()
axs[4].plot(exits)
#axs[4].plot(asset_returns_rescaled)
plt.show()

fig, axs = plt.subplots(nrows=3, ncols=1)
axs[0].plot(w)
axs[0].set_title('Portfolio with FTRL-qN')
axs[1].plot(ww)
axs[1].set_title('Portfolio with FTRL-qN + Restart')
axs[2].plot(capital_algo, label='with FTRL-qN')
axs[2].plot(capital_uniform, label='with Uniform Portfolio')
axs[2].plot(capital_algo_r, label='with FTRL-qN + Restart')
axs[2].set_title('Capital Evolution')
axs[2].legend()
plt.show()

fig, axs = plt.subplots(nrows=4, ncols=1)
axs[0].plot(asset_prices)
axs[1].plot(w)
axs[2].plot(capital_algo, label='with FTRL-qN')
axs[2].plot(capital_uniform, label='with Uniform Portfolio')
axs[2].set_title('Capital Evolution')
axs[2].legend()
axs[3].plot(castd)
plt.show()
'''

fig, axs = plt.subplots(nrows=5, ncols=1)
axs[0].plot(asset_prices)
axs[0].set_title('Asset Prices')
axs[1].plot(past_w)
axs[1].set_title('Portfolio with FTRL-qN')
axs[2].plot(past_ww)
axs[2].set_title('Portfolio with FTRL-qN + Restart (capital)')
axs[3].plot(past_www)
axs[3].set_title('Portfolio with FTRL-qN + Restart (portfolio)')
axs[4].plot(capital_algo, label='with FTRL-qN')
axs[4].plot(capital_uniform, label='with Uniform Portfolio')
axs[4].plot(capital_algo_r, label='with FTRL-qN + Restart (capital)')
axs[4].plot(capital_algo_www, label='with FTRL-qN + Restart (portfolio)')
axs[4].set_title('Capital Evolution')
axs[4].legend()
#axs[5].plot(castd)
#axs[5].set_title('Evolution of the cumulative STD for Restart (portfolio)')
plt.show()


