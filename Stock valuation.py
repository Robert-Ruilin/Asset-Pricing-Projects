"""
Stock Valuation

Part 1: Performance Measurement
Risk_Factors.xlsx contains monthly observations of the risk-free rate and the three Fama–French risk factors, 
all expressed as a percentage. These observations cover the ten-year period from Jan 2004 through Dec 2013.

Using excess returns for the ten industry portfolios, calculate the following performance metrics:
1. Sharpe ratio
2. Sortino ratio (with risk-free rate as target)
3. Jensen's alpha
4. Three-factor alpha
Create a table showing the performance metrics for the ten industry portfolios. 
Also plot your results as a bar chart for each performance metric. 
Briefly explain the economic significance of each performance metric.

Part 2: Minimum-Variance Frontier Revisited
Use the monthly returns of the ten industry portfolios to generate the minimum-variance frontier 
without short sales, using Monte Carlo simulation. 

Without short sales, portfolio weights will be limited to the range [0, 1]. 
Randomly draw each element of w, the vector of portfolio weights, from the uniform distribution in the range [0, 1]. 
Divide w by the sum of the portfolio weights, to ensure that the portfolio weights sum to one.

Use the normalized w to calculate the mean return and standard deviation of return. 
Repeat this process until you have at least 100,000 obervations. 
Plot the points with mean return on the vertical axis and standard deviation of return 
on the horizontal axis to show the minimum-variance frontier.

Repeat this entire process by simulating 1/w using the standard uniform distribution: i.e., 
take the reciprocal of the random draw from the standard uniform distribution as the portfolio weight. 
Plot your results to show the minimum-variance frontier on a separate graph.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data_idt = pd.read_excel('Industry_Portfolios.xlsx', index_col=0, header=0)
data_mkt = pd.read_excel('Market_Portfolio.xlsx', index_col=0, header=0)
data_risk = pd.read_excel('Risk_Factors.xlsx', index_col=0, header=0)

'''---Part 1---'''
#Sharp ratio
Rf = data_risk[['Rf']]
Rf.columns = [data_idt.columns[0]]
for i in range(1,len(data_idt.columns)):
    Rf.insert(i,data_idt.columns[i],Rf[[data_idt.columns[0]]])
Risk_premium = data_idt - Rf
Mean_Risk_premium = Risk_premium.mean()
Var_Risk_premium = Risk_premium.var()
Std_Risk_premium = Risk_premium.std()
Sharp_ratio = pd.DataFrame(Mean_Risk_premium/Std_Risk_premium,index=data_idt.columns.T,columns=['Sharp ratio'])
print(Sharp_ratio)

#Sortino ratio
Rf_1 = Rf.copy()
for i in range(0,len(data_idt.index)):
    for j in range(0,len(data_idt.columns)):
        if data_idt.iloc[i,j] >= Rf_1.iloc[i,j]:
            Rf_1.iloc[i,j] = np.nan
Risk_premium_d = data_idt - Rf_1
Semivar_Risk_premium = (Risk_premium_d**2).sum()/len(Risk_premium_d)
Sortino_ratio = pd.DataFrame(Mean_Risk_premium/np.sqrt(Semivar_Risk_premium),index=data_idt.columns.T,columns=['Sortino ratio'])
print(Sortino_ratio)

#Jensen's alpha
Market_Risk_premium = data_risk[['Rm-Rf']]
regression_mm = LinearRegression().fit(Market_Risk_premium,Risk_premium)
Jensen_alpha = pd.DataFrame(regression_mm.intercept_,index=data_idt.columns.T,columns=['Jensen\'s alpha'])
print(Jensen_alpha)

#Three-factor alpha
Dependent_factors = data_risk[['Rm-Rf','SMB','HML']]
regression_tfm = LinearRegression().fit(Dependent_factors,Risk_premium)
Three_factor_alpha = pd.DataFrame(regression_tfm.intercept_,index=data_idt.columns.T,columns=['Three-factor alpha'])
print(Three_factor_alpha)

#Performance metrics for the ten industry portfolios
Performance_metrics = pd.concat([Sharp_ratio.T,Sortino_ratio.T,Jensen_alpha.T,Three_factor_alpha.T]).T
print(Performance_metrics)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=[15,12])
fig.suptitle('Performance metrics for ten industry portfolios',fontsize=30)
ax1.bar(Sharp_ratio.index,Sharp_ratio['Sharp ratio'],0.5)
ax1.set_xlabel('Industry')
ax1.set_ylabel('Sharp ratio')
ax2.bar(Sortino_ratio.index,Sortino_ratio['Sortino ratio'],0.5)
ax2.set_xlabel('Industry')
ax2.set_ylabel('Sortino ratio')
ax3.bar(Jensen_alpha.index,Jensen_alpha['Jensen\'s alpha'],0.5)
ax3.set_xlabel('Industry')
ax3.set_ylabel('Jensen\'s alpha')
ax4.bar(Three_factor_alpha.index,Three_factor_alpha['Three-factor alpha'],0.5)
ax4.set_xlabel('Industry')
ax4.set_ylabel('Three-factor alpha')
ax1.set_title('Sharp ratio for ten industry portfolios',fontsize=15)
ax2.set_title('Sortino ratio for ten industry portfolios',fontsize=15)
ax3.set_title('Jensen\'s alpha for ten industry portfolios',fontsize=15)
ax4.set_title('Three factor alpha for ten industry portfolios',fontsize=15)
ax1.grid(axis='y')
ax2.grid(axis='y')
ax3.grid(axis='y')
ax4.grid(axis='y')

'''---Part 2---'''
#Monte Carlo simulation for minimum variance frontier    
num_p = 100000
all_weights = np.zeros((num_p,len(data_idt.columns)))
Rt_portfolio = np.zeros(num_p)
Std_portfolio = np.zeros(num_p)
for i in range(num_p):
    weights = np.random.uniform(0,1,len(data_idt.columns))
    weights = weights/np.sum(weights)
    all_weights[i,:] = weights
    Rt_portfolio[i] = np.sum(data_idt.mean()*weights)
    Std_portfolio[i] = np.sqrt(np.dot(weights.T,np.dot(data_idt.cov(),weights)))

fig,ax = plt.subplots(figsize=(12,10))
plt.scatter(Std_portfolio,Rt_portfolio,c='b')
ax.set_xlim(0,10)
ax.set_ylim(0,2)
plt.title('Minimum-variance frontier generated by Monte Carlo simulation',fontsize=20)
plt.xlabel('Volatility (sigma %)')
plt.ylabel('Portfolio Return (%)')

for i in range(num_p):
    weights = np.random.uniform(0,1,len(data_idt.columns))
    weights = weights/np.sum(weights)
    weights_1 = 1/weights
    weights_1 = weights_1/np.sum(weights_1)
    all_weights[i,:] = weights_1
    Rt_portfolio[i] = np.sum(data_idt.mean()*weights_1)
    Std_portfolio[i] = np.sqrt(np.dot(weights_1.T,np.dot(data_idt.cov(),weights_1)))

fig,ax = plt.subplots(figsize=(12,10))
plt.scatter(Std_portfolio,Rt_portfolio,c=Rt_portfolio,cmap='viridis')
ax.set_xlim(0,10)
ax.set_ylim(0,2)
plt.title('Minimum-variance frontier generated by Monte Carlo simulation-2',fontsize=20)
plt.colorbar()
plt.xlabel('Volatility (sigma %)')
plt.ylabel('Portfolio Return (%)')
    
    





