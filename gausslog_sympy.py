# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:35:18 2016

@author: quentinpeter

This script uses sympy to get the least square fit  formula of a gaussian

Copyright (C) 2016  Quentin Peter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
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