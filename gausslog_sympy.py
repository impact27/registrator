# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:35:18 2016

@author: quentinpeter
"""
#Gauss = exp (C0-1/(2*C2)*(x-C1)**2)
#Variance=C2
#Mean = C1
#The least square method leads to the following with:
#Di=sum(D*x**i)
#Xi=sum(x**i)

import sympy
from sympy import Symbol
C=[Symbol('C'+str(x)) for x in range(3)]
D=[Symbol('D'+str(x)) for x in range(3)]
X=[Symbol('X'+str(x)) for x in range(5)]

e=[2*C[2]*(C[0]*X[0+x]-D[0+x])-(C[1]**2*X[0+x]-2*C[1]*X[1+x]+X[2+x])
     for x in range(3)]
result=sympy.solvers.solve(e,C)
#%%