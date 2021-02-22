#!/usr/bin/python
'''
ECE 172A, Homework 2 Robot Traversal
Author: regreer@ucsd.edu
For use by UCSD ECE 172A students only.
'''
#!/usr/bin/python2.7 --> #!/usr/bin/python
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.warnings.filterwarnings(
    'ignore', category=np.VisibleDeprecationWarning  # remove warnings
)

initial_loc = np.array([0, 0])
final_loc = np.array([100, 100])
sigma = np.array([[20, 0], [0, 20]])
mu = np.array([[10, 30], [30, 10]])

# def getDif(gx,gy,e,x,y):
#     def getAbs(r,x,y):
#         print(x,y)
#         return abs(r[x][y])
    
#     return getAbs(gx,x,y) > e or getAbs(gy,x,y) > e
# def findMin()
def gradiantDescent(a,e,gx,gy,pos,ax):
    x,y = pos
    iters = 3000
    positions = []
    i = 0
    # while getDif(gx,gy,e,x,y):
    while i < iters:
        # print(i+1)
        i += 1
        # if gx[x][y] 
        # S = int(a*gx[x][y])
        # x,y = x - S, y - S
        
        # if len(positions) > 1000: positions.pop(0)
        ax.plot(-x,-y,2,marker=(6,2,0), color='red', ms=10)
        if abs(x) < 100 and abs(y) < 100:
            if abs(x) < 100: x = int(x+ np.sign(a*gx[x][y]))
            if abs(y) < 100: y = int(y + np.sign(a*gy[x][y]))
        # else: 
        #     print("hi")
        #     positions.append((x,y))
        #     break
        
        if (x,y) in positions: 
            x += 1
        positions.append((x,y))
        gy, gx = np.gradient(z)
        # norm= np.linalg.norm(np.array((dy, dx)), axis=0) 
        # gy = gy/ norm
        # gx = gx / norm
        if x == -100 and y == -100: break
    return positions


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
ax.quiver(x,y,dx,dy)
print(gradiantDescent(1,0.0001,dx,dy,(0,0),ax))
ax.set_ylabel('y')
# ax.set_zlabel('z')
ax.set_title('2D Contour')
plt.show()



# plt.plot(x,z)
# plt.show()

# ii
v,u= np.gradient(z)
norm = np.linalg.norm(np.array((u,v)), axis=0) 
u = -u/ norm
v = -v / norm
ax.quiver(x,y,u,v, units='xy', scale=.5, color='gray')

# plt.plot(2,2, marker=(6,2,0), color='blue', ms=10)



