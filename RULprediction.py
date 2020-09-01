# -*- coding: utf-8 -*-
"""
Created on Sat May  9 00:47:11 2020
DESCRIPTION: RUL Prediction - Particle Filter
Version: Simple 1.00
@author: hasan
"""

# %%
""" DATA PROCESS """
# read data
import scipy.io as sio
data = sio.loadmat('TESTLED10.mat')
locals().update(data)
del data

# time - unit information
time_int = 21
time_unit = 'Hours'
import numpy as np
time = np.zeros((len(led),1))
time[0,:] = 0
for i in range (1,len(led)):
    time[i,:] = time[i-1,:]+time_int

# create dataframe with given data
import pandas as pd
LED = pd.DataFrame({'timeunit': time[:,0],
                   'data': led[:,0]})

# normalized data
LED ['dataNormalized'] = LED['data']/100;


# plot check
LED.plot(x='timeunit',y='data',figsize=(12,8),title="original data").autoscale(axis='x',tight=True);
#LED['dataNormalized'].plot(figsize=(12,8),title="normalized data").autoscale(axis='x',tight=True);

# save data
LED.to_csv('LED.csv')


# %%
""" PARAMETER DEFINATION """

# data separation
split_point = 11
train = LED['dataNormalized'].iloc[:split_point]
test = LED['dataNormalized'].iloc[split_point:]

# for particle filter

thres = 0.7;                    # threshold - critical value

# model parameter
# probability parameters of initial distribution, p x q
# p: num. of unknown param
# q: num. of probability pa
ParamNumber = 3
ParamIn = pd.DataFrame({'paramname': ["x", "b", "s"],
                   'lowerlimit': [0.9,0,0.01],
                   'upperlimit': [1.1,0.05,0.1]})

# number of particle
numparticle = 6500;
# significance level for C.I. and P.I.
signiLevel = 5;                  


""" DATA PROCESS """

# generate particle for the Params
import numpy as np
xresul = np.random.uniform(ParamIn['lowerlimit'].iloc[0],ParamIn['upperlimit'].iloc[0],numparticle)
bresul = np.random.uniform(ParamIn['lowerlimit'].iloc[1],ParamIn['upperlimit'].iloc[1],numparticle)
sresul = np.random.uniform(ParamIn['lowerlimit'].iloc[2],ParamIn['upperlimit'].iloc[2],numparticle)

# reshape
xresul = np.resize (xresul,(1,len(xresul)))
bresul = np.resize (bresul,(1,len(bresul)))
sresul = np.resize (sresul,(1,len(sresul)))


param = np.concatenate((xresul,bresul,sresul), axis=0)

# %%
k1 = len(train)-1; 
# Update Process or Prognosis
k=0;               

if train.values[-1] - train.values[0]<0:
    cofec=-1; 
else: 
    cofec=1;





# %%
""" MAIN CALCULATION """
from scipy.stats import norm
import random as rnd 

while (min(xresul[k,:]*cofec))<(thres*cofec):
    # print (k)
    k = k+1
    # step 1. prediction (prior)
    parampredi = param
    
    x = parampredi[0,:]
    b = parampredi[1,:]
    s = parampredi[2,:]
    
    # reshape
    x = np.resize (x,(1,len(x)))
    b = np.resize (b,(1,len(b)))
    s = np.resize (s,(1,len(s)))

    
    parampredi[0,:] = np.exp(-1*b*time_int)*x
    
    if k<=k1:
        # step 2. update (likelihood)
        likel = norm.pdf(train.values[k], parampredi[0,:], parampredi[2,:]);
        likel = np.resize (likel,(1,len(likel)))
        
        # step 3. resampling
        cdf = np.cumsum(likel)/np.sum(likel)
        cdf = np.resize (cdf,(1,len(cdf)))
        
        for i in range (0, numparticle):
            # print (i)
            u = rnd.random()
            loca = np.where(cdf[0,:]>=u)
            loca = np.transpose(np.asarray(loca))
            param [:,i] = np.transpose(parampredi[:,loca[0]])
    else:
        param = parampredi

    tempx = param[0,:]
    tempx = np.resize (tempx,(1,len(tempx)))
    xresul = np.concatenate((xresul,tempx), axis=0)
    
    tempb = param[1,:]
    tempb = np.resize (tempb,(1,len(tempb)))
    bresul = np.concatenate((bresul,tempb), axis=0)
    
    temps = param[2,:]
    temps = np.resize (temps,(1,len(temps)))
    sresul = np.concatenate((sresul,temps), axis=0)

    if k>k1:
        tempx = np.random.normal(param[0,:],param[2,:])
        tempx = np.resize (tempx,(1,len(tempx)))
        xresul = np.concatenate((xresul,tempx), axis=0)



# %%
""" Post-Processing """
time_p = np.zeros((k,1))
time_p[0,:] = 0
for i in range (1,k):
    time_p[i,:] = time_p[i-1,:]+time_int

perceValue = [50, signiLevel, 100-signiLevel]

RUL = np.zeros((1,numparticle)) 
for i in range (0,numparticle):
    loca2 = np.where((xresul[:,i]*cofec)>=(thres*cofec))
    loca2 = np.transpose(np.asarray(loca2))
    RUL[0,i] = time_p[loca2[0]]-time_p[k1]


# calculate the pecentile from RUL
RULPerce = np.percentile(RUL[0,:],perceValue)

# %%
""" GRAPH AND PLOT """

# plot RUL histogram and save
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.hist(RUL[0,:], bins = 30)
plt.xlabel('RUL ('+time_unit+')')
plt.xlim(np.min(RUL),np.max(RUL))
title = 'Percentiles of RUL at : ' + str(int(time_p.ravel()[k1])) + time_unit
plt.title(title)
plt.savefig('RULhist.pdf', bbox_inches="tight", dpi = 300)
plt.show()



# prediction
prediction = xresul[:,k]*100
xlen1 = time

xlen2 = np.zeros((len(prediction),1))
xlen2[0,:] = 0
for i in range (1,len(prediction)):
    xlen2[i,:] = xlen2[i-1,:]+time_int

# plot and save
plt.style.use("ggplot")
plt.plot(xlen2[split_point:int(((RULPerce[0]+time_p.ravel()[k1])/time_int)+1),0], prediction[split_point:int(((RULPerce[0]+time_p.ravel()[k1])/time_int)+1)])
plt.plot(xlen1[:,0],LED['data'].iloc[:])
plt.xlabel('Time ('+time_unit+')')
plt.xlabel('Light Output (%)')
plt.title('RUL Prediction')
plt.legend(['RUL estimated by the PF-based approach', 'True'])
plt.savefig('FinalFIg.pdf', bbox_inches="tight", dpi = 300)
plt.show()
