# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:17:28 2020

@author: viola
"""
import math
from enum import Enum
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import bisect
from scipy import optimize
from scipy.stats import norm
import pandas as pd
from scipy.misc import derivative

        
def pdePricerX(S0, r, q, vol, NX, NT, w, trade):
    # set up pde grid
    mu = r - q
    T = trade.expiry
    X0 = math.log(S0)
    #vol0 = lv.LV(0, S0)
    srange = 5 * vol * math.sqrt(T)
    maxX = X0 + (mu - vol * vol * 0.5)*T + srange
    minX = X0 - (mu - vol * vol * 0.5)*T - srange
    dt = T / (NT-1)
    dx = (maxX - minX) / (NX-1)
    # set up spot grid
    xGrid = np.array([minX + i*dx for i in range(NX)])
    
    # initialize the payoff
    ps = np.array([trade.payoff(math.exp(x)) for x in xGrid])
    # backward induction
    for j in range(1, NT):
        # set up the matrix, for LV we need to update it for each iteration
        M = np.zeros((NX, NX))
        D = np.zeros((NX, NX))
        for i in range(1, NX - 1):
            #vol = lv.LV(j*dt, math.exp(xGrid[i]))
            M[i, i - 1] = (mu - vol * vol / 2.0) / 2.0 / dx - vol * vol / 2 / dx / dx
            M[i, i] = r + vol * vol / dx / dx
            M[i, i + 1] = -(mu - vol * vol / 2.0) / 2.0 / dx - vol * vol / 2 / dx / dx
            D[i, i] = 1.0
        # the first row and last row depends on the boundary condition
        M[0, 0], M[NX - 1, NX - 1] = 1.0, 1.0
        rhsM = (D - dt * M) * w + (1 - w) * np.identity(NX)
        lhsM = w * np.identity(NX) + (D + dt * M) * (1 - w)
        inv = np.linalg.inv(lhsM)

        ps = rhsM.dot(ps)
        ps[0] = dt*math.exp(-r*j*dt) * trade.payoff(math.exp(xGrid[0])) # discounted payoff
        ps[NX-1] = dt*math.exp(-r*j*dt) * trade.payoff(math.exp(xGrid[NX-1]))
        ps = inv.dot(ps)
    # linear interpolate the price
    
    return np.interp(X0, xGrid, ps),np.interp(math.log(S0+100*dx), xGrid, ps),np.interp(math.log(S0-100*dx), xGrid, ps),100*dx



def pdeExplicitPricer(S0, r, q, vol, NS, NT, trade):
    
    # set up pde grid
    mu, T = r - q, trade.expiry
    srange = 5 * vol * math.sqrt(T)
    maxS = S0 * math.exp((mu - vol * vol * 0.5)*T + srange)
    minS = S0 * math.exp((mu - vol * vol * 0.5)*T - srange)
    dt, dS = T / (NT-1), (maxS - minS) / (NS-1)
    sGrid = np.array([minS + i*dS for i in range(NS)]) # set up spot grid
    ps = np.array([trade.payoff(s) for s in sGrid]) # initialize the payoff
     # set up the matrix, for BS the matrix does not change, for LV we need to update it for each iteration
    a, b = mu/2.0/dS, vol * vol / dS / dS
    M = np.zeros((NS, NS))
    D = np.zeros((NS, NS))
    for i in range (1, NS-1):
        M[i, i-1] = a*sGrid[i] - b*sGrid[i]*sGrid[i]/2.0
        M[i, i], D[i,i] = r + b * sGrid[i] * sGrid[i], 1.0
        M[i, i+1] = -a * sGrid[i] - b*sGrid[i]*sGrid[i]/2.0
     # the first row and last row depends on the boundary condition - we use Dirichlet here
    M[0,0], M[NS-1, NS-1] = 1.0, 1.0
    M = D - dt * M
    for j in range(1, NT): # backward induction
        ps = M.dot(ps) # Euler explicit
        ps[0] = math.exp(-r*j*dt) * trade.payoff(sGrid[0]) # discounted payoff
        ps[NS-1] = math.exp(-r*j*dt) * trade.payoff(sGrid[NS-1])
    return np.interp(S0, sGrid, ps) # linear interpolate the price at S0


def pdeImplicitPricer(S0, r, q, vol, NS, NT, trade):
 # set up pde grid
    mu, T = r - q, trade.expiry
    srange = 5 * vol * math.sqrt(T)
    maxS = S0 * math.exp((mu - vol * vol * 0.5)*T + srange)
    minS = S0 * math.exp((mu - vol * vol * 0.5)*T - srange)
    dt, dS = T / (NT-1), (maxS - minS) / (NS-1)
    sGrid = np.array([minS + i*dS for i in range(NS)]) # set up spot grid
    ps = np.array([trade.payoff(s) for s in sGrid]) # initialize the payoff
    # set up the matrix, for BS the matrix does not change, for LV we need to update it for each iteration
    a, b = mu/2.0/dS, vol * vol / dS / dS
    M = np.zeros((NS, NS))
    D = np.zeros((NS, NS))
    for i in range (1, NS-1):
        M[i, i-1] = a*sGrid[i] - b*sGrid[i]*sGrid[i]/2.0
        M[i, i], D[i,i] = r + b * sGrid[i] * sGrid[i], 1.0
        M[i, i+1] = -a * sGrid[i] - b*sGrid[i]*sGrid[i]/2.0
    # the first row and last row depends on the boundary condition - we use Dirichlet here
    M[0,0], M[NS-1, NS-1] = 1.0, 1.0
    
    # only different from explicit at backward induction HERE
    M = np.linalg.inv(D + dt * M)
    for j in range(1, NT):
        ps[0] = dt*math.exp(-r*j*dt) * trade.payoff(sGrid[0]) # discounted payoff
        ps[NS-1] = dt*math.exp(-r*j*dt) * trade.payoff(sGrid[NS-1])
        ps = M.dot(ps) # Euler implicit
    return np.interp(S0, sGrid, ps) # linear interpolate the price at S0