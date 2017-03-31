# Martin Nisser 19/03/2017
# Script for Path planning of N=3 satellites forming a Cubic-P Bravais lattice, using Dario Izzo's paper Autonomous and Distributed Motion Planning for Satellite Swarm (2006)


from sympy import *
from sympy.physics.mechanics import *
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.ion() # interactive plotting on
# Imports for visualization
from pydy.viz.shapes import Cube, Cylinder
from pydy.viz.visualization_frame import VisualizationFrame
from pydy.viz.scene import Scene

import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from random import random

plt.ion() # interactive plotting on

# Unlike EM_Dynamics, you don't need the generate_ode_function part as you already have your equations of motion (just velocity as a function of position) and just have to integrate them. In contrast, EM_dynamics uses kinematics to generate the EoMs as accelerations and velocities, and then integrates that to get the state vector: position and velocity.

## Global Variables
N = 8 # number of satellites
L = 0.1 # side length of Cube
dt = 0.2 # integration time step
T = 150 # Total simulation time
scale = 0.05 # used to scale the co-ordinates for the swarm initialization

# Choose swarm variables b,c,d,k_a, k_d
init_uniform = True
init_oneSided = not init_uniform

if (init_uniform == True ) and (init_oneSided == False):
    # Initialization 1
    b = 0.8
    d = 0.5
    k_a = 0.1 * L
    k_d = 0.1 * L


    Ag1Xo=  -10 * scale * random()
    Ag1Yo = -10 * scale * random()
    Ag1Zo = -10 * scale * random()

    Ag2Xo =  10 * scale * random()
    Ag2Yo = -10 * scale * random()
    Ag2Zo = -10 * scale * random()

    Ag3Xo = -10 * scale * random()
    Ag3Yo =  10 * scale * random()
    Ag3Zo = -10 * scale * random()

    Ag4Xo = -10 * scale * random()
    Ag4Yo = -10 * scale * random()
    Ag4Zo =  10 * scale * random()

    Ag5Xo =  10 * scale * random()
    Ag5Yo =  10 * scale * random()
    Ag5Zo = -10 * scale * random()

    Ag6Xo =  10 * scale * random()
    Ag6Yo = -10 * scale * random()
    Ag6Zo =  10 * scale * random()

    Ag7Xo = -10 * scale * random()
    Ag7Yo =  10 * scale * random()
    Ag7Zo =  10 * scale * random()

    Ag8Xo =  10 * scale * random()
    Ag8Yo =  10 * scale * random()
    Ag8Zo =  10 * scale * random()
elif (init_oneSided == True) and (init_uniform == False):
    # Initialization 2
    b = 0.8
    d = 0.5
    k_a = 0.1 * L
    k_d = 0.1 * L

    Ag1Xo = 10 * scale
    Ag1Yo = 10 * scale
    Ag1Zo = 10 * scale

    Ag2Xo = 6 * scale
    Ag2Yo = 6 * scale
    Ag2Zo = 8 * scale

    Ag3Xo = 12 * scale
    Ag3Yo = 17 * scale
    Ag3Zo = 14 * scale

    Ag4Xo = 15 * scale
    Ag4Yo = 10 * scale
    Ag4Zo = 12 * scale

    Ag5Xo = 10 * scale
    Ag5Yo = 8  * scale
    Ag5Zo = 11 * scale

    Ag6Xo = 9  * scale
    Ag6Yo = 12 * scale
    Ag6Zo = 8  * scale

    Ag7Xo = 9  * scale
    Ag7Yo = 7  * scale
    Ag7Zo = 11 * scale

    Ag8Xo = 7  * scale
    Ag8Yo = 12 * scale
    Ag8Zo = 17 * scale
else:
    print('\n You must select one and only one initialization')

c = 0.25 * ( b * ( np.exp(-L**2 / k_a) + 2*np.exp(-2*L**2 / k_a) + np.exp(-3*L**2 / k_a) ) - d * ( exp(-L**2 / k_d) + 2*np.exp(-2*L**2 / k_d) + np.exp(-3*L**2 / k_d) ) )

# Desired Destination "Dest" Positions (Nx3) of N satellites
Dest = np.zeros((N,3))
Dest[1,:] =  [L,0,0]
Dest[2,:] =  [0,L,0]
Dest[3,:] =  [0,0,L]
Dest[4,:] =  [L,L,0]
Dest[5,:] =  [L,0,L]
Dest[6,:] =  [0,L,L]
Dest[7,:] =  [L,L,L]
#print('Targets=\n',Dest)

P = np.zeros((N,3))

P[:,:] = [[Ag1Xo,Ag1Yo,Ag1Zo],
          [Ag2Xo,Ag2Yo,Ag2Zo],
          [Ag3Xo,Ag3Yo,Ag3Zo],
          [Ag4Xo,Ag4Yo,Ag4Zo],
          [Ag5Xo,Ag5Yo,Ag5Zo],
          [Ag6Xo,Ag6Yo,Ag6Zo],
          [Ag7Xo,Ag7Yo,Ag7Zo],
          [Ag8Xo,Ag8Yo,Ag8Zo]]
#print('Initialized Pos=\n',P)

## System dynamics is a sum of 3 behaviours/velocities: Gather, avoid and dock.
# Desired velocity= V_Gather + V_Avoid + V_Dock

# Initialization: Initial Velocity (Nx3) of N satellites
V = np.zeros((N,3))

# Gather velocity term
def Gather(P,Dest):
    toDest = Dest - P # vector pointing from sat i to dest j
    fun_G = 1
    return c * fun_G * toDest

# Docking velocity term
def Dock(P,Dest):
    toDest = Dest - P # vector pointing from sat i to dest j
    toDest_mag = np.linalg.norm(toDest)
    fun_D = exp(-toDest_mag**2 / k_d)
    return d * fun_D * toDest

# Avoid velocity term
def Avoid(Pi,Pj):
    toSat = Pi - Pj # note reversal; vector points from sat j to sat i
    toSat_mag = np.linalg.norm(toSat)
    fun_A = exp(-toSat_mag**2 / k_a)
    return b * fun_A * toSat

def ComputeVelocity(P):
    v_G = np.zeros(3)
    v_D = np.zeros(3)
    v_A = np.zeros(3)
    dv = np.zeros((N,3)) # N rows will store store vA+v_Ds+v_G from each target dests/sat
    V = np.zeros((N,3)) # N rows will store the desired Vel for the N satellites
    for i in range(N):
        for j in range(N):
            v_G = Gather(P[i,:] , Dest[j,:])
            v_D = Dock(P[i,:] , Dest[j,:])
            v_A = Avoid(P[i,:] , P[j,:])
            dv[j,:] = v_G + v_D + v_A
        V[i,:] = np.sum(dv,axis=0) # sum along x axis, in order to sum Xs, then Ys then Zs
    return V

# Integrate Velocity for Positions
steps = int(T/dt + 1)
t = np.linspace(0,T,steps)
Pos = np.zeros(((N,3,len(t))))
Pos[:,:,0] = P
Vel = np.zeros(((N,3,len(t))))

for i in range (steps):
    # find current velocity based on current position
    Vel[:,:,i] = ComputeVelocity(Pos[:,:,i])
    if (i == (steps-1)): # i.e. if on last iteration
        break # then don't try to calculate next position, as it's out of bounds
    # next position = current position + current velocity * dt
    Pos[:,:,i+1] = Pos[:,:,i] +  Vel[:,:,i] * dt
    #print('vel=' , Vel[:,:,i])
    #print('pos=' , Pos[:,:,i])


def Plotting(Pos): # Plots 2D x and y positions, with time on z axis
    plt.close('all')
    fig=plt.figure()
    ax = p3.Axes3D(fig)

    ax.plot3D(Pos[0,0,:],Pos[0,1,:],t)
    ax.plot3D(Pos[1,0,:],Pos[1,1,:],t)
    ax.plot3D(Pos[2,0,:],Pos[2,1,:],t)
    ax.plot3D(Pos[3,0,:],Pos[3,1,:],t)
    ax.plot3D(Pos[4,0,:],Pos[4,1,:],t)
    ax.plot3D(Pos[5,0,:],Pos[5,1,:],t)
    ax.plot3D(Pos[6,0,:],Pos[6,1,:],t)
    ax.plot3D(Pos[7,0,:],Pos[7,1,:],t)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

print('pos=' , Pos[:,:,-1]) # get positions at last time step
Plotting(Pos)

def plotting3DStatic(Pos): # plots 3D, whole trajectory
    fig=plt.figure()
    ax = p3.Axes3D(fig)

    # Plot trajectories
    ax.plot3D(Pos[0,0,:],Pos[0,1,:],Pos[0,2,:],c='b')
    ax.plot3D(Pos[1,0,:],Pos[1,1,:],Pos[1,2,:],c='g')
    ax.plot3D(Pos[2,0,:],Pos[2,1,:],Pos[2,2,:],c='r')
    ax.plot3D(Pos[3,0,:],Pos[3,1,:],Pos[3,2,:],c='c')
    ax.plot3D(Pos[4,0,:],Pos[4,1,:],Pos[4,2,:],c='m')
    ax.plot3D(Pos[5,0,:],Pos[5,1,:],Pos[5,2,:],c='y')
    ax.plot3D(Pos[6,0,:],Pos[6,1,:],Pos[6,2,:],c='burlywood')
    ax.plot3D(Pos[7,0,:],Pos[7,1,:],Pos[7,2,:],c='chartreuse')

    # Mark initial positions
    ax.scatter(Pos[0,0,0],Pos[0,1,0],Pos[0,2,0],c='b',marker='*',s=200)
    ax.scatter(Pos[1,0,0],Pos[1,1,0],Pos[1,2,0],c='g',marker='*',s=200)
    ax.scatter(Pos[2,0,0],Pos[2,1,0],Pos[2,2,0],c='r',marker='*',s=200)
    ax.scatter(Pos[3,0,0],Pos[3,1,0],Pos[3,2,0],c='c',marker='*',s=200)
    ax.scatter(Pos[4,0,0],Pos[4,1,0],Pos[4,2,0],c='m',marker='*',s=200)
    ax.scatter(Pos[5,0,0],Pos[5,1,0],Pos[5,2,0],c='y',marker='*',s=200)
    ax.scatter(Pos[6,0,0],Pos[6,1,0],Pos[6,2,0],c='burlywood',marker='*',s=200)
    ax.scatter(Pos[7,0,0],Pos[7,1,0],Pos[7,2,0],c='chartreuse',marker='*',s=200)
    # Mark final positions
    ax.scatter(Pos[0,0,-1],Pos[0,1,-1],Pos[0,2,-1],c='b',marker='s',s=200)
    ax.scatter(Pos[1,0,-1],Pos[1,1,-1],Pos[1,2,-1],c='g',marker='s',s=200)
    ax.scatter(Pos[2,0,-1],Pos[2,1,-1],Pos[2,2,-1],c='r',marker='s',s=200)
    ax.scatter(Pos[3,0,-1],Pos[3,1,-1],Pos[3,2,-1],c='c',marker='s',s=200)
    ax.scatter(Pos[4,0,-1],Pos[4,1,-1],Pos[4,2,-1],c='m',marker='s',s=200)
    ax.scatter(Pos[5,0,-1],Pos[5,1,-1],Pos[5,2,-1],c='y',marker='s',s=200)
    ax.scatter(Pos[6,0,-1],Pos[6,1,-1],Pos[6,2,-1],c='burlywood',marker='s',s=200)
    ax.scatter(Pos[7,0,-1],Pos[7,1,-1],Pos[7,2,-1],c='chartreuse',marker='s',s=200)
    # Mark Destinations, i.e. target positions
    '''
    ax.scatter(Dest[0,0],Dest[0,1],Dest[0,2],c='k', marker='*')
    ax.scatter(Dest[1,0],Dest[1,1],Dest[1,2],c='k', marker='*')
    ax.scatter(Dest[2,0],Dest[2,1],Dest[2,2],c='k', marker='*')
    ax.scatter(Dest[3,0],Dest[3,1],Dest[3,2],c='k', marker='*')
    ax.scatter(Dest[4,0],Dest[4,1],Dest[4,2],c='k', marker='*')
    ax.scatter(Dest[5,0],Dest[5,1],Dest[5,2],c='k', marker='*')
    ax.scatter(Dest[6,0],Dest[6,1],Dest[6,2],c='k', marker='*')
    ax.scatter(Dest[7,0],Dest[7,1],Dest[7,2],c='k', marker='*')
    '''

    ax.set_xlim3d([min(Ag1Xo,Ag2Xo,Ag3Xo,Ag4Xo,Ag5Xo,Ag6Xo,Ag7Xo,Ag8Xo,0), max(Ag1Xo,Ag2Xo,Ag3Xo,Ag4Xo,Ag5Xo,Ag6Xo,Ag7Xo,Ag8Xo,0)])
    ax.set_xlabel('X (m)')
    ax.set_ylim3d([min(Ag1Yo,Ag2Yo,Ag3Yo,Ag4Yo,Ag5Yo,Ag6Yo,Ag7Yo,Ag8Yo,0), max(Ag1Yo,Ag2Yo,Ag3Yo,Ag4Yo,Ag5Yo,Ag6Yo,Ag7Yo,Ag8Yo,0)])
    ax.set_ylabel('Y (m)')
    ax.set_zlim3d([min(Ag1Zo,Ag2Zo,Ag3Zo,Ag4Zo,Ag5Zo,Ag6Zo,Ag7Zo,Ag8Zo,0), max(Ag1Zo,Ag2Zo,Ag3Zo,Ag4Zo,Ag5Zo,Ag6Zo,Ag7Zo,Ag8Zo,0)])
    ax.set_zlabel('Z (m)')
    plt.show()

plotting3DStatic(Pos)

##############################
#### Animation Functions #####
##############################

def makeTrajectory(n,agent):
    x, y, z = Pos[agent,0,:],Pos[agent,1,:],Pos[agent,2,:]
    Agent = np.vstack((x, y, z))
    return Agent

def update_lines(num, dataLines, lines) :
    for line, data in zip(lines, dataLines) :
        line.set_data(data[0:2, num-1:num])
        line.set_3d_properties(data[2,num-1:num])
    return lines

def plottingAnimation():

    # Attach 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    n = 50
    data =  [makeTrajectory(n,0)]
    data2 = [makeTrajectory(n,1)]
    data3 = [makeTrajectory(n,2)]
    data4 = [makeTrajectory(n,3)]
    data5 = [makeTrajectory(n,4)]
    data6 = [makeTrajectory(n,5)]
    data7 = [makeTrajectory(n,6)]
    data8 = [makeTrajectory(n,7)]

    lines =  [ax.plot(data[0][0,0:1],  data[0][1,0:1],  data[0][2,0:1],  's',markersize=10)[0]]
    lines2 = [ax.plot(data2[0][0,0:1], data2[0][1,0:1], data2[0][2,0:1], 's',markersize=10)[0]]
    lines3 = [ax.plot(data3[0][0,0:1], data3[0][1,0:1], data3[0][2,0:1], 's',markersize=10)[0]]
    lines4 = [ax.plot(data4[0][0,0:1], data4[0][1,0:1], data4[0][2,0:1], 's',markersize=10)[0]]
    lines5 = [ax.plot(data5[0][0,0:1], data5[0][1,0:1], data5[0][2,0:1], 's',markersize=10)[0]]
    lines6 = [ax.plot(data6[0][0,0:1], data6[0][1,0:1], data6[0][2,0:1], 's',markersize=10)[0]]
    lines7 = [ax.plot(data7[0][0,0:1], data7[0][1,0:1], data7[0][2,0:1], 's',markersize=10)[0]]
    lines8 = [ax.plot(data8[0][0,0:1], data8[0][1,0:1], data8[0][2,0:1], 's',markersize=10)[0]]

    # Set the axes properties
    ax.set_xlim3d([min(Ag1Xo,Ag2Xo,Ag3Xo,Ag4Xo,Ag5Xo,Ag6Xo,Ag7Xo,Ag8Xo,0), max(Ag1Xo,Ag2Xo,Ag3Xo,Ag4Xo,Ag5Xo,Ag6Xo,Ag7Xo,Ag8Xo,0)])
    ax.set_xlabel('X')
    ax.set_ylim3d([min(Ag1Yo,Ag2Yo,Ag3Yo,Ag4Yo,Ag5Yo,Ag6Yo,Ag7Yo,Ag8Yo,0), max(Ag1Yo,Ag2Yo,Ag3Yo,Ag4Yo,Ag5Yo,Ag6Yo,Ag7Yo,Ag8Yo,0)])
    ax.set_ylabel('Y')
    ax.set_zlim3d([min(Ag1Zo,Ag2Zo,Ag3Zo,Ag4Zo,Ag5Zo,Ag6Zo,Ag7Zo,Ag8Zo,0), max(Ag1Zo,Ag2Zo,Ag3Zo,Ag4Zo,Ag5Zo,Ag6Zo,Ag7Zo,Ag8Zo,0)])
    ax.set_zlabel('Z')

    # Creating the Animation object
    # Calling several throws an exception, but it still works
    ani =  animation.FuncAnimation(fig, update_lines, n, fargs=(data,  lines),  interval=50, blit=False)
    ani2 = animation.FuncAnimation(fig, update_lines, n, fargs=(data2, lines2), interval=50, blit=False)
    ani3 = animation.FuncAnimation(fig, update_lines, n, fargs=(data3, lines3), interval=50, blit=False)
    ani4 = animation.FuncAnimation(fig, update_lines, n, fargs=(data4, lines4), interval=50, blit=False)
    ani5 = animation.FuncAnimation(fig, update_lines, n, fargs=(data5, lines5), interval=50, blit=False)
    ani6 = animation.FuncAnimation(fig, update_lines, n, fargs=(data6, lines6), interval=50, blit=False)
    ani7 = animation.FuncAnimation(fig, update_lines, n, fargs=(data7, lines7), interval=50, blit=False)
    ani8 = animation.FuncAnimation(fig, update_lines, n, fargs=(data8, lines8), interval=50, blit=False)

    plt.show()
    plt.off() # throws "AttributeError", but won't animate otherwise

plottingAnimation()

#############################
#### Animation Finished #####
#############################


#TODO Simulation: should be straight forward to size some cube shapes according such that they touch when in targte positions, and tie them to frames fixed in Inertial frame so they never rotate.


# TODO Not necessary, but could be good exercise: formulate and symbolically integrate a differential equation dy/dt + p(t) y = g(t), as this e function is just the solution to y for an initial value problem.


## "exploiting the gravitational environment"
# To account for gravitational effects in orbit, we modify the simple dynamics to account for and exploit geodesics, by using the Clohessy-Wiltshire (or Hill's) equations. The CW equations give the motion of a chaser craft w.r.t. a target that is itself on a circular orbit around a central body (like Earth).
# Can't access ref 7 but concise explanation, with details on including height above earth are found here:
# https://en.wikipedia.org/wiki/Clohessy-Wiltshire_equations
# and the ABCD matrices described in paper are found here, bottom of page 1:
# http://www.ae.utexas.edu/courses/ase366k/cw_equations.pdf
# TODO Text in the ABCD matrices
# TODO Update Gather velocity term (Nx3), V_Gather




## Control
# Optional, not required: the Equilibrium shaping defines (desired) velocities for each satellite at any position, given the positions of all its neighbours: we could just implement these "forced" velocities without any regard for disturbances, inputs or inertia, by assuming each satellite can track this velocity perfectly. This is very different from a description of actual real dynamics as done in EM, where kinematic constraints are introduced and where you then need to give inputs/forces, or atleast initial velocities, in order to make things move.
# Implement just 1 type, e.g. Sliding Mode Control. Ask Dario for some help.
