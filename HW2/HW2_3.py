'''
ECE 172A, Homework 2 Robot Kinematics
Author: regreer@ucsd.edu
For use by UCSD ECE 172A students only.
'''

import numpy as np
import matplotlib.pyplot as plt
import math

def forwardKinematics(theta0, theta1, theta2, l0, l1, l2):
    J1_x = l0*math.cos(theta0)
    J1_y = l0*math.sin(theta0)
    J2_x = l1*math.cos(theta0+theta1) + J1_x
    J2_y = l1*math.sin(theta0+theta1) + J1_y
    E_x = l2*math.cos(theta0+theta1+theta2)+J2_x
    E_y = l2*math.sin(theta0+theta1+theta2)+J2_y
    return J1_x,J1_y,J2_x,J2_y,E_x,E_y

    '''
	This function is supposed to implement inverse kinematics for a robot arm
	with 3 links constrained to move in 2-D. The comments will walk you through
	the algorithm for the Jacobian Method for inverse kinematics.

	INPUTS:
	l0, l1, l2: lengths of the robot links
	x_e_target, y_e_target: Desired final position of the end effector 

	OUTPUTS:
	theta0_target, theta1_target, theta2_target: Joint angles of the robot that
	take the end effector to [x_e_target,y_e_target]
	'''
 


def inverseKinematics(l0, l1, l2, x_e_target, y_e_target):
    def make_jacobian(theta0,theta1,theta2,l0,l1,l2):
        jacobian = np.zeros((2,3))
        jacobian[0,0] = -l0*math.sin(theta0)-l1*math.sin(theta0+theta1)-l2*math.sin(theta0+theta1+theta2)
        jacobian[0,1] = -l1*math.sin(theta0+theta1) - l2*math.sin(theta0+theta1+theta2)
        jacobian[0,2] = -l2*math.sin(theta0+theta1+theta2)
        jacobian[1,0] = l0*math.cos(theta0)+l1*math.cos(theta0+theta1)+l2*math.cos(theta0+theta1+theta2)
        jacobian[1,1] = l1*math.cos(theta0+theta1)+l2*math.sin(theta0+theta1+theta2)
        jacobian[1,2] = l2*math.cos(theta0+theta1+theta2)
        return jacobian

    # Initialize for the plots:
    end_effector_positions = []

    # Initialize the thetas to some value
    theta0,theta1,theta2 = math.pi/3, 0,0
    # Obtain end effector position x_e, y_e for current thetas:
    # HINT: use your ForwardKinematics function
    a,b,c,d, x_e, y_e = forwardKinematics(theta0,theta1,theta2,l0,l1,l2)
    end_effector_positions.append([x_e,y_e])
    # Replace the '1' with a condition that checks if your estimated [x_e,y_e] is close to [x_e_target,y_e_target]
    while math.sqrt( (x_e_target-x_e)**2 + (y_e_target-y_e)**2 ) > 0.1:
        
        # Calculate the Jacobian matrix for current values of theta
        # HINT: write a function for doing this
        jacobian = make_jacobian(theta0,theta1,theta2,l0,l1,l2)
        # Calculate the pseudo-inverse of the jacobian (HINT: numpy pinv())
        inverse_jacobian = np.linalg.pinv(jacobian)
        # Update the values of the thetas by a small step
        gx = x_e_target-x_e
        gy = y_e_target-y_e
        
        a=np.array([[gx],[gy]])
        g_theta=0.1*np.matmul(inverse_jacobian,a)
        
        theta0 = theta0 + g_theta[0]
        theta1 = theta1 + g_theta[1]
        theta2 = theta2 + g_theta[2]
        # print(theta0,theta1,theta2)
        print(math.sqrt( (x_e_target-x_e)**2 + (y_e_target-y_e)**2 ))
        # Obtain end effector position x_e, y_e for the updated thetas:
        a,b,c,d,x_e,y_e = forwardKinematics(theta0,theta1,theta2,l0,l1,l2)
        # If you would like to visualize the iterations, draw the robot using drawRobot.

        # Save end effector positions for the plot:
        end_effector_positions.append([x_e,y_e])
        # Plot the final robot pose
        # Plot the end effector position through the iterations
    return end_effector_positions, theta0[0],theta1[0],theta2[0]


def drawRobot(x_1, y_1, x_2, y_2, x_e, y_e):
    x_0, y_0 = 0, 0
    plt.plot([x_0, x_1, x_2, x_e], [y_0, y_1, y_2, y_e], lw=4.5)
    plt.scatter([x_0, x_1, x_2, x_e], [y_0, y_1, y_2, y_e], color='r')
    plt.show()

# a,b,c,d,e,f = forwardKinematics(math.pi/4,-math.pi/4,math.pi/4,3,5,2)
# drawRobot(a,b,c,d,e,f)

myList, t1,t2,t3 = inverseKinematics(10,10,10,6,12)
for i in range(len(myList)):
    plt.scatter(myList[i][0],myList[i][1])
    # plt.pause(0.5)
print(t1 * 180 / math.pi,t2 * 180 / math.pi,t3 * 180 / math.pi)
plt.show()


# a,b,c,d,e,f = forwardKinematics(t1,t2,t3,10,10,10)
# drawRobot(a,b,c,d,e,f)