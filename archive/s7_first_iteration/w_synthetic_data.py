import VB_FTRL 
import numpy as np
import matplotlib.pyplot as plt
from price import prix, returns,rescale

d = 10  # Number of assets
T = 350  # Number of time rounds

asset_returns = prix(d,T) 
# asset_returns = returns(d,T)
asset_prices = np.multiply.accumulate(asset_returns,0)    
asset_returns_rescaled = rescale(asset_returns - 1,method="minmax",treshold = 0.02)


capital_algo = np.zeros(T)
capital_algo[0] = 1

capital_uniform = np.zeros(T)
capital_uniform[0] = 1

algo = VB_FTRL.VB_FTRL(d,mu = 1, lambda_val= 1, S = 15)
pastw = [algo.w]

for tau in range(T-1):
    w = algo.w
    capital_algo[tau+1] = capital_algo[tau]*np.sum(w*asset_returns[tau])
    capital_uniform[tau+1] = capital_uniform[tau]*np.sum(asset_returns[tau]/d)
    # algo.w = np.ones(d)/d
    algo.update(asset_returns_rescaled[tau])
    algo.w[algo.w < 0] = 0.01
    algo.w = algo.w / np.linalg.norm(algo.w,1)
    pastw.append(algo.w)
    # print(f"market :{capital_uniform[tau+1]}, algo : {capital_algo[tau+1]}, {capital_algo[tau+1] > capital_uniform[tau+1]}")
    

# fig, axs = plt.subplots(nrows=4, ncols=1)
# axs[0].plot(asset_prices)
# axs[1].plot(pastw)
# axs[2].plot(capital_algo)
# axs[2].plot(capital_uniform)
# axs[3].plot(asset_returns_rescaled)
# plt.show()

####################################################################
fig, ax = plt.subplots(nrows=3, ncols=1)
plt.rcParams['figure.figsize'] = [18, 14]
####################################################################
ax[0].set_title("Prix des actifs")
ax[0].set_ylabel("prix")
ax[0].plot(asset_prices)
####################################################################
ax[1].set_title("Poids du portefeuille de l'algorithme")
ax[1].plot(pastw)
####################################################################
ax[2].set_title("Valeurs du portefeuille de l'algorithme et du portefeuille équiréparti")
ax[2].plot(capital_algo, label = 'Algorithme')
ax[2].plot(capital_uniform, label = 'Portefeuille équiréparti')
ax[2].legend()

plt.tight_layout()
plt.show()


    

