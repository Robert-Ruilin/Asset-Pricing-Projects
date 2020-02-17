"""
Linear Factor Models

Market_Portfolio.xlsx contains monthly nominal (net) returns for the market portfolio, 
expressed as a percentage. These returns cover the ten-year period from Jan 2004 through Dec 2013.

Assume that the risk-free rate is 0.13% per month. Regress the monthly excess returns for 
each of the ten industry portfolios on the monthly excess returns for the market portfolio, 
so as to estimate the intercept coefficient (alpha) and slope coefficient (beta) for each of 
the ten industry portfolios. Create a table showing the intercept and slope coefficients for 
the ten industry portfolios. Briefly explain the economic significance of the intercept 
and slope coefficients.

Calculate the mean monthly return for each of the ten industry portfolios, as well as 
the market portfolio. Regress the mean monthly returns of the ten industry portfolios 
and the market portfolio on the corresponding betas (by construction, the market portfolio 
has beta of one). This will give you the slope and intercept coefficients for the SML. 
(Warning: the results may be very different from what you would expect!)

Using the estimated slope and intercept coefficients, plot the SML in the range of beta 
from zero to two on the horizontal axis. Also plot the positions of the ten industry portfolios 
and the market portfolio. (You are NOT required to label the individual portfolios.) 
Briefly explain the economic significance of the SML.

Please submit your results (including graphs and qualitative discussion of economic significance) 
and programming code to the submission folder for Homework 2 before the start of the lecture 
on Thursday, 31 October. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data_mkt = pd.read_excel('Market_Portfolio.xlsx',index_col=0,header=0)/100
data_idt = pd.read_excel('Industry_Portfolios.xlsx',index_col=0,header=0)/100

#Regress the excess returns of industry portfolios on the excess returns for market portfolio
rf = 0.0013
y1 = data_idt.iloc[:] - rf
x1 = data_mkt.iloc[:] - rf
rg_exrt = LinearRegression().fit(x1,y1)
idt_alpha = rg_exrt.intercept_
idt_beta = rg_exrt.coef_
rg_coefficient = pd.DataFrame(np.concatenate((idt_alpha.reshape(1,10),idt_beta.reshape(1,10))),
                                       index = ['intercept coefficient','slope coefficient'],
                                       columns = data_idt.columns)
print(rg_coefficient)

#Regress the mean returns of portfolios on the betas
data_mkt_mean = np.array(data_mkt.mean())
data_idt_mean = np.array(data_idt.mean())
mkt_beta = np.ones((1,1))
mean_rt = pd.DataFrame(np.concatenate((data_idt_mean,data_mkt_mean)),columns=['mean rt'])
beta = pd.DataFrame(np.concatenate((idt_beta,mkt_beta)),columns=['beta'])

y2 = mean_rt
x2 = beta
rg_SML = LinearRegression().fit(x2,y2)
intercept_SML = rg_SML.intercept_
slope_SML = rg_SML.coef_
print(intercept_SML)
print(slope_SML)

#Plot Security Market Line
fig,ax=plt.subplots(1,1,figsize=(8,6))
plt.xlim(0,2)
plt.ylim(0,0.015)
f = lambda x: slope_SML[0][0]*x + intercept_SML[0]
x = np.array([0,2])
plt.plot(x,f(x),c='b',label='Security Market Line')
plt.scatter(beta,mean_rt,c='r',label='Industry Portfolios')
plt.scatter(mkt_beta,data_mkt_mean,c='g',label='Market Portfolio')
plt.ylabel('Mean Return (R)')
plt.xlabel('Beta (β)')
plt.title('Security Market Line')
plt.legend()
plt.show()





    


