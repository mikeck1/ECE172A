'''
ECE 172A, Homework 2 Robot Traversal
Author: regreer@ucsd.edu
For use by UCSD ECE 172A students only.
'''

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.warnings.filterwarnings(
    'ignore', category=np.VisibleDeprecationWarning  # remove warnings
)

initial_loc = np.array([0, 0])
final_loc = np.array([100, 100])
sigma = np.array([[50, 0], [0, 50]])
mu = np.array([[60, 50], [10, 40]])

def getDif(gx,gy,e,x,y):
    def getAbs(r,x,y):
        return abs(int(r[x,y]))
    
    return getAbs(gx,x,y) > e or getAbs(gy,x,y) > e
    
def gradiantDescent(a,e,gx,gy,pos):
    x,y = pos
    while getDif(gx,gy,e,x,y):
        S = a*[gx[x,y]]
        pos = np.subtract(pos,S)
        x,y = pos
        plt.plot(x,y,2,marker=(6,2,0), color='green', ms=10)
        if x<0: x = int(x-a*np.sign(gx[x,y]))
        if y < 0: y = int(y - a*np.sign(gy[x,y]))

        gy, gx = np.gradient(z)
        pos = x, y

def f(x, y):
    return ((final_loc[0]-x)**2 + (final_loc[1]-y)**2)/20000 + 10000*(1/(2*np.pi*np.linalg.det(sigma)))*np.exp(-.5*(np.matmul(np.array([x-mu[0, 0], y-mu[0, 1]]), np.matmul(np.linalg.pinv(sigma), np.atleast_2d(np.array([x-mu[0, 0], y-mu[0, 1]])).T)))[0]) + 10000*(1/(2*np.pi*np.linalg.det(sigma)))*np.exp(-.5*(np.matmul(np.array([x-mu[1, 0], y-mu[1, 1]]), np.matmul(np.linalg.pinv(sigma), np.array([x-mu[1, 0], y-mu[1, 1]])))))


x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
z = f(x[:, None], y[None, :])

fig = plt.figure()
ax = plt.axes()
ax.contour(x, y, z, 100)
ax.set_xlabel('x')
dy, dx = np.gradient(z)
# ax.quiver(x,y,dx,dy)
ax.set_ylabel('y')
# ax.set_zlabel('z')
ax.set_title('2D Contour')
plt.show()

plt.plot(x,z)
plt.show()

# ii
v,u= np.gradient(z)
norm = np.linalg.norm(np.array((u,v)), axis=0) 
u = -u/ norm
v = -v / norm
ax.quiver(x,y,u,v, units='xy', scale=.5, color='gray')

plt.plot(2,2, marker=(6,2,0), color='blue', ms=10)



