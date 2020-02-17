"""
Multi-Period Asset Pricing

Suppose that consumption growth has a lognormal distribution with the possibility of rare disasters:
ln g̃ = 0.02 + 0.02*ϵ̃  + ν̃ 
Here epsilon is a standard normal random variable, while nu is an independent random variable 
that has value of either zero (with probability of 98.3%) or ln(0.65) (with probability of 1.7%). 
Simulate epsilon with 10,000 random draws from a standard normal distribution, 
and simulate nu with 10,000 random draws from a standard uniform distribution. 

Part 1: Hansen–Jagannathan Bound
Use the simulated distribution of consumption growth to calculate the pricing kernel for power utility:
M̃ = 0.99*g̃**(−γ) , for gamma in the range [1,4]. 
Calculate the mean and standard deviation of the pricing kernel for all values of gamma. 
Plot the ratio SD(M)/E(M) (on the vertical axis) vs gamma (on the horizontal axis). 
Take note of the smallest value of gamma for which SD(M)/E(M) > 0.4 
(i.e., for which the Hansen–Jagannathan bound is satisfied). 
Briefly explain the economic significance of this result.

Part 2: Price-Dividend Ratio
Use the simulated distribution of consumption growth to find the (constant) price-dividend ratio 
for the first equity claim, for gamma in the range [1, 7]:
P(1)/D = E[0.99*g̃**(1−γ)], Plot P(1)/D (on the vertical axis) vs gamma (on the horizontal axis).

Part 3: Equity Premium
Use the simulated distribution of consumption growth to find the expected market return, 
for gamma in the range [1, 7]:
E[R̃(m)] = D/P(1)*E[g̃]
Use the simulated distribution of consumption growth to find the risk-free rate, 
for gamma in the range [1,7]:
R(f) = 1/E[0.99*g̃**(−γ)]
Plot the equity premium (on the vertical axis) vs gamma (on the horizontal axis).

Please submit your results (including graphs and qualitative discussion of economic significance) 
and programming code to the submission folder for Homework 4 
before the start of the lecture on Thursday, 14 November.
"""
import numpy as np
import matplotlib.pyplot as plt

num_c = 10000
consumption_growth = np.zeros(num_c)
for i in range(num_c):
    mu, sigma = 0,1
    epsilon = np.random.standard_normal()
    prob_nu = np.random.uniform(0,1)
    if prob_nu <0.017:
        nu = np.log(0.65)
    else:
        nu = 0
    consumption_growth[i] = np.exp(0.02 + 0.02*epsilon + nu)

'''---Part 1:Hansen-Jagannathon Bound---'''
num_g = 201
delta = 0.99
gamma1 = np.linspace(1,4,num_g)
HJ_bound = np.zeros(num_g)
for i in range(num_g):
    HJ_bound[i] = np.std(delta*consumption_growth**(-gamma1[i]))/np.mean(delta*consumption_growth**(-gamma1[i]))
SM_idx = np.min(np.argwhere(HJ_bound>0.4))
SM_value = HJ_bound[np.min(np.argwhere(HJ_bound >0.4))]
gamma1_ns, HJ_bound_ns = gamma1[:SM_idx], HJ_bound[:SM_idx]
gamma1_s, HJ_bound_s = gamma1[SM_idx:], HJ_bound[SM_idx:]

fig,ax1 = plt.subplots(figsize=(10,8))
ax1.plot(gamma1_ns,HJ_bound_ns,c='b',ls='--',label='unsatisfied HJ bound which is below Sharp ratio')
ax1.plot(gamma1_s,HJ_bound_s, c='b',label='satisfied HJ bound which is above Sharp ratio')
plt.scatter(gamma1[SM_idx],SM_value,c='r',label='Smallest value of gamma for which SD(M)/E(M)>0.4')
plt.xlabel('gamma')
plt.ylabel('SD(M)/E(M)')
plt.title('SD(M)/E(M)',fontsize=20)
plt.legend(loc='upper left')
plt.show()

fig,ax = plt.subplots(figsize=(10,8))
ax.plot(gamma1_s,HJ_bound_s,c='b')
plt.xlabel('gamma')
plt.ylabel('SD(M)/E(M)')
plt.title('Hansen-Jagannathon Bound',fontsize=20)

'''---Part 2:Price-Dividend Ratio---'''
gamma2 = np.linspace(1,7,num_g)
PD_ratio = np.zeros(num_g)
for i in range(num_g):
    PD_ratio[i] = np.mean(delta*consumption_growth**(1-gamma2[i]))

fig,ax2 = plt.subplots(figsize=(10,8))
ax2.plot(gamma2,PD_ratio,c='b')
plt.xlabel('gamma')
plt.ylabel('P(1)/D')
plt.title('Price-Dividend Ratio',fontsize=20)

'''---Part 3:Equity Premium---'''
mkt_rt, rf, equity_premium = np.zeros(num_g), np.zeros(num_g), np.zeros(num_g)
for i in range(num_g):
    mkt_rt[i] = np.mean(consumption_growth)/PD_ratio[i]
    rf[i] = 1/np.mean(delta*consumption_growth**(-gamma2[i]))
    equity_premium[i] = np.log(mkt_rt[i])-np.log(rf[i])

fig,ax3 = plt.subplots(figsize=(10,8))
ax3.plot(gamma2,equity_premium,c='b')
plt.xlabel('gamma')
plt.ylabel('Equity Premium')
plt.title('Equity Premium',fontsize=20)
    
