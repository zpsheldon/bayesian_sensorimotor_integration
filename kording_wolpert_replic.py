#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 22:37:49 2018

@author: Zach Sheldon
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Model/Result Replication from Kording & Wolpert, 2004: Bayesian Integration in Sensorimotor Learning

# task: Subjects were instructed to move a cursor from one end of a screen to the other (20 cm away from them)
# once the cursor moved (and the task began), the cursor was shifted laterally to the right by some amount drawn
# from a prior distribution with mean 1 cm and std 0.5 cm. Halfway through the movement (at 10 cm), the subjects
# either received complete feedback of the cursor's location (condition 1), partially distorted feedback (condition 2),
# a higher degree of distorted feedback (condition 3), or no feedback at all (condition 4). They then had to move the cursor
# to the target 20 cm away, and in condition 1, they received feedback of where their cursor actually ended up.

# subjects were trained on 1000 trials, and then 1000 trials were used for testing/analysis. 
# Data is available at crcns.org under DREAM dataset
# This analysis used simulated data based on the figures from the paper

# model 1: full compensation for visual feedback
#           -  average lateral deviation at the end of the movement should be 0 for all conditions
# model 2 (bayesian): optimally use info about prior and level of uncertainty to estimate deviation
#           - estimated lateral deviation should move towards mean of prior as uncertainty increases
# model 3: learn a mapping between final visual feedback and an estimate (requires feedback in all conditions)
#           - compensate for deviation (which is 1 cm on average) independent of uncertainty, thus all conditions should 
#           exhibit same slope as condition 1, which is the only one with feedback at the end of the movement

##########################################################################################################

# plot shifts
plt.figure(0)
def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp( - ((x - mean) / standard_deviation) ** 2)

x = np.random.normal(10, 5, size=10000)
bin_heights, bin_borders,_ = plt.hist(np.random.normal(1.0, 0.5, 1000), density=True)

bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])

x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), 'k')
plt.title('Prior Distribution - Lateral Shifts')
plt.xlabel('Lateral Shift (cm)')
plt.xlim([-0.5, 2.5])
plt.ylabel('Probability')
plt.show()
#plt.savefig('lateral_shifts.png')

# simulated data - condition 1

# lateral shift
np.random.seed(1234)
lat_shift_cond1 = np.linspace(0.0, 2.0, 12)
epsilon_cond1 = np.random.normal(0.025, 0.075, 12)

# deviation from target
final_deviation_cond1 = np.linspace(0.0, 2.0, 12)
def deviation_cond1(x):
    return (0.25 * x - 0.25) 

for i in range(0, len(final_deviation_cond1)):
    final_deviation_cond1[i] = deviation_cond1(final_deviation_cond1[i]) + epsilon_cond1[i]
    

# x_estimated = (sigma_prior**2 / sigma_prior**2 + sigma_sensed**2)(x_sensed) + 
#               (sigma_sensed**2 / sigma_prior**2 + sigma_sensed**2)(1 cm)
# where sigma_prior**2 = (0.5)**2 and x_sensed is the deviation from target

# bayesian model
sigma_prior = 0.5
X1 = lat_shift_cond1
y1 = final_deviation_cond1
def bayesian_model(X, sigma_sensed_squared):
    b1 = (sigma_prior**2 / (sigma_prior**2 + sigma_sensed_squared))
    b0 = (sigma_sensed_squared / (sigma_prior**2 + sigma_sensed_squared))
    return b1*X + b0

# linear reg.
param_opt, param_cov = curve_fit(bayesian_model, X1, y1)
y_pred_cond1 = bayesian_model(X1, *param_opt) - 1.0

# plot results
fig_cond1 = plt.figure()
plt.plot([0, 0, 0], 'b', linestyle='dotted', label='Model 1')
plt.plot([0, 1, 2],[-1, 0, 1], 'k', linestyle='dashed', label='Model 3')
plt.plot(lat_shift_cond1, final_deviation_cond1, 'm', marker='o', markevery=1, label='Data')
plt.plot(X1, y_pred_cond1, 'k', label='Bayesian fit')
plt.xlim([0,2])
plt.ylim([-1,1])
#plt.legend()
plt.xlabel('True lateral shift (cm)')
plt.ylabel('Deviation from target (cm)')
plt.title('Condition 1 - No Visual Noise') 
#fig_cond1.savefig('condition1.png')

#############################################################################################

# simulated data - condition 2

# lateral shift
np.random.seed(12345)
lat_shift_cond2 = np.linspace(0.0, 2.0, 12)
epsilon_cond2 = np.random.normal(0.025, 0.075, 12)

# deviation from target
final_deviation_cond2 = np.linspace(0.0, 2.0, 12)
def deviation_cond2(x):
    return (0.45 * x - 0.45) 

for i in range(0, len(final_deviation_cond2)):
    final_deviation_cond2[i] = deviation_cond2(final_deviation_cond2[i]) + epsilon_cond2[i]

# bayesian model
X2 = lat_shift_cond2
y2 = final_deviation_cond2
param_opt2, param_cov2 = curve_fit(bayesian_model, X2, y2)
y_pred_cond2 = bayesian_model(X2, *param_opt2) - 1.0
    
# plot results
fig_cond2 = plt.figure()
plt.plot([0, 0, 0], 'b', linestyle='dotted', label='Model 1')
plt.plot([0, 1, 2],[-1, 0, 1], 'k', linestyle='dashed', label='Model 3')
plt.plot(lat_shift_cond1, final_deviation_cond2, 'c', marker='o', markevery=1, label='Data')
plt.plot(X2, y_pred_cond2, 'k', label='Bayesian fit')
plt.xlim([0,2])
plt.ylim([-1,1])
#plt.legend()
plt.xlabel('True lateral shift (cm)')
plt.ylabel('Deviation from target (cm)')
plt.title('Condition 2 - Medium Visual Noise')
#fig_cond2.savefig('condition2.png')

#############################################################################################
    
# simulated data - condition 3

# lateral shift
np.random.seed(123456)
lat_shift_cond3 = np.linspace(0.0, 2.0, 12)
epsilon_cond3 = np.random.normal(0.025, 0.075, 12)

# deviation from target
final_deviation_cond3 = np.linspace(0.0, 2.0, 12)
def deviation_cond3(x):
    return (0.6 * x - 0.6) 

for i in range(0, len(final_deviation_cond3)):
    final_deviation_cond3[i] = deviation_cond3(final_deviation_cond3[i]) + epsilon_cond3[i]

# bayesian model
X3 = lat_shift_cond3
y3 = final_deviation_cond3
param_opt3, param_cov3 = curve_fit(bayesian_model, X3, y3)  
y_pred_cond3 = bayesian_model(X3, *param_opt3) - 1.0
    
# plot results
fig_cond3 = plt.figure()
plt.plot([0, 0, 0], 'b', linestyle='dotted', label='Model 1')
plt.plot([0, 1, 2],[-1, 0, 1], 'k', linestyle='dashed', label='Model 3')
plt.plot(lat_shift_cond3, final_deviation_cond3, 'g', marker='o', markevery=1, label='Data')
plt.plot(X3, y_pred_cond3, 'k', label='Bayesian fit')
plt.xlim([0,2])
plt.ylim([-1,1])
#plt.legend()
plt.xlabel('True lateral shift (cm)')
plt.ylabel('Deviation from target (cm)')
plt.title('Condition 3 - Large Visual Noise')
#fig_cond3.savefig('condition3.png')

#############################################################################################
    
# simulated data - condition 4

# lateral shift
np.random.seed(1234567)
lat_shift_cond4 = np.linspace(0.0, 2.0, 12)
epsilon_cond4 = np.random.normal(0.025, 0.075, 12)

# deviation from target
final_deviation_cond4 = np.linspace(0.0, 2.0, 12)
def deviation_cond4(x):
    return (0.95 * x - 0.95) 

for i in range(0, len(final_deviation_cond4)):
    final_deviation_cond4[i] = deviation_cond4(final_deviation_cond4[i]) + epsilon_cond4[i]
    
# bayesian model
X4 = lat_shift_cond4
y4 = final_deviation_cond4
param_opt4, param_cov4 = curve_fit(bayesian_model, X4, y4)  
y_pred_cond4 = bayesian_model(X4, *param_opt4) - 1.0

# plot results
fig_cond4 = plt.figure()
plt.plot([0, 0, 0], 'b', linestyle='dotted', label='Model 1')
plt.plot([0, 1, 2],[-1, 0, 1], 'k', linestyle='dashed', label='Model 3')
plt.plot(lat_shift_cond4, final_deviation_cond4, 'y', marker='o', markevery=1, label='Data')
plt.plot(X4, y_pred_cond4, 'k', label='Bayesian fit')
plt.xlim([0,2])
plt.ylim([-1,1])
#plt.legend()
plt.xlabel('True lateral shift (cm)')
plt.ylabel('Deviation from target (cm)')
plt.title('Condition 4 - No Feedback')
#fig_cond4.savefig('condition4.png')

#############################################################################################

# simulated data - slope plot
fig_slopes = plt.figure()
cond1, cond2, cond3, cond4 = plt.bar(np.arange(1,5), [0.25, 0.45, 0.6, 0.95], width=0.35);
cond1.set_facecolor('m')
cond2.set_facecolor('c')
cond3.set_facecolor('g')
cond4.set_facecolor('y')
plt.ylabel('Slope')
plt.xlabel('Feedback Condition')
plt.xticks(np.arange(1,5), ['Condition 1', 'Condition 2', 'Condition 3', 'Condition 4'])
plt.ylim([0, 1])
plt.title('Slope - Simulated Data')
#fig_slopes.savefig('slopes.png')

# calculate slope of Bayesian fit
slope_cond1 = (sigma_prior**2) / (sigma_prior**2 + param_opt)
slope_cond2 = (sigma_prior**2) / (sigma_prior**2 + param_opt2)
slope_cond3 = (sigma_prior**2) / (sigma_prior**2 + param_opt3)
slope_cond4 = (sigma_prior**2) / (sigma_prior**2 + param_opt4)

# plot slopes of Bayesian fits
fig_slopes_fit = plt.figure()
cond1, cond2, cond3, cond4 = plt.bar(np.arange(1,5), [slope_cond1[0], slope_cond2[0], slope_cond3[0], slope_cond4[0]], width=0.35);
cond1.set_facecolor('m')
cond2.set_facecolor('c')
cond3.set_facecolor('g')
cond4.set_facecolor('y')
plt.ylabel('Slope')
plt.xlabel('Feedback Condition')
plt.xticks(np.arange(1,5), ['Condition 1', 'Condition 2', 'Condition 3', 'Condition 4'])
plt.ylim([0, 1])
plt.title('Slope - Bayesian Fits')
#fig_slopes_fit.savefig('slopes_fit.png')

#############################################################################################

# bias against gain for each of the linear fits

# we expect no deviation from the target if lateral shift is at mean of the prior (1 cm),
# this predicts that the sum of the slope and the offset (y-intercept) should be 0

# plot
fig_bias = plt.figure()
plt.plot([0.25, 0.45, 0.6, 0.95], [-0.25, -0.45, -0.6, -0.95], 'k')
plt.yticks([-0.2, -0.4, -0.6, -0.8, -1.0])
plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0])
plt.xlabel('Slope (cm)')
plt.ylabel('Bias (cm)')
#fig_bias.savefig('bias_slope.png')

#############################################################################################

# MSE

# model 1: x_estimated = x_sensed, therefore MSE = sigma_sensed**2
model1_mse_cond1 = mean_squared_error(final_deviation_cond1, np.zeros(12))
model1_mse_cond2 = mean_squared_error(final_deviation_cond2, np.zeros(12))
model1_mse_cond3 = mean_squared_error(final_deviation_cond3, np.zeros(12))
model1_mse_cond4 = mean_squared_error(final_deviation_cond4, np.zeros(12))
model1_mse_ave = (model1_mse_cond1 + model1_mse_cond2 + model1_mse_cond3 + model1_mse_cond4) / (4.0)

# model 2: MSE = (sigma_sensed**2 * sigma_prior**2) / (sigma_sensed**2 + sigma_prior**2), thus is always lower
model2_mse_cond1 = mean_squared_error(final_deviation_cond1, y_pred_cond1)
model2_mse_cond2 = mean_squared_error(final_deviation_cond2, y_pred_cond2)
model2_mse_cond3 = mean_squared_error(final_deviation_cond3, y_pred_cond3)
model2_mse_cond4 = mean_squared_error(final_deviation_cond4, y_pred_cond4)
model2_mse_ave = (model2_mse_cond1 + model2_mse_cond2 + model2_mse_cond3 + model2_mse_cond4) / (4.0)

# model 3: predicts same average response towards mean of the prior for all conditions, 
# thus is discounted by the data

# plot
fig_mse = plt.figure()
model1_msevals = [model1_mse_cond1, model1_mse_cond2, model1_mse_cond3, model1_mse_cond4]
model2_msevals = [model2_mse_cond1, model2_mse_cond2, model2_mse_cond3, model2_mse_cond4]
ind = np.arange(1, 5)
width = 0.35
rects1 = plt.bar(ind, model1_msevals, width, color='r', label='Model 1')
rect2 = plt.bar(ind+width, model2_msevals, width, color='b', label='Model 2')
plt.ylabel('MSE')
plt.xticks(ind+width, ('Cond. 1', 'Cond. 2', 'Cond. 3', 'Cond. 4'))
plt.legend()
#fig_mse.savefig('mse_bars.png')

#############################################################################################
