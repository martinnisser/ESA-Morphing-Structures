# EM dynamics
# Created by Martin Nisser, 7th February 2017
# Script to generate and solve the equations of motion for two EM-driven EM satellites
# For help see PyDy Tutorials: https://github.com/pydy/pydy-tutorial-human-standing http://www.pydy.org/examples/double_pendulum.html

import time as myTime
start_time = myTime.time()

# Imports for the dynamics and plotting
from sympy import *
from sympy.physics.mechanics import *
from numpy import deg2rad, rad2deg, array, zeros, linspace,empty_like,pi,ones,sqrt
import numpy as np
from scipy.integrate import odeint
from pydy.codegen.ode_function_generators import generate_ode_function
import matplotlib.pyplot as plt
plt.ion() # interactive plotting on
# Imports for visualization
from pydy.viz.shapes import Cube, Cylinder
from pydy.viz.visualization_frame import VisualizationFrame
from pydy.viz.scene import Scene

from scipy.integrate import ode

# How many cubes
numCubes = 2

## Generalized co-ordinates x, y, theta1/I, theta2/J
# q1z, q2z are counter clockwise (right hand rule), x components giving the angles of Cube CoMs
qx, qy, qz, q1x, q1y, q1z, q2z = dynamicsymbols('qx qy qz q1x q1y q1z q2z')
qxd, qyd, qzd, q1xd, q1yd, q1zd, q2zd = dynamicsymbols('qx qy qz q1x q1y q1z q2z', 1) # skipped in human tut.
# Derivatives of Generalized Co-ordinates
ux, uy, uz, u1x, u1y, u1z, u2z = dynamicsymbols('ux uy uz u1x u1y u1z u2z')
uxd, uyd, uzd, u1xd, u1yd, u1zd, u2zd = dynamicsymbols('ux uy uz u1x u1y u1z u2z', 1) # skipped in human tut.
# Parameter to set later along with mass and inertia; s is cube side length
s = symbols('s')

# Reference Frames
N = ReferenceFrame('N') # Inertial frame
# While all subsequent frames/cubes are described by 1 angle w.r.t. previous frame/cube in the chain, I1 must be described by all 3 angles w.r.t. inertial so the whole system can move in 3d. This means choosing what convention to use, e.g. Euler ZXZ, quaternion, etc. Now, if you initialize all 3 angles at 0, then your frame is aligned with the N frame, and all conventions give you the identity matrix since they are all just different ways of putting together sins and cosines. Even after initializations, our objective here is ultimately just the simulation, so we don't really care what the angles are, but we WOULD care if we want to initialize cube 1 in a certain configuration: then we need to think and figure out what angles (e.g. z,x,z in ZXZ convention) actually get us that configuration. For different conventions, a given 3d orientation will have completely different rotation matrices, as they are made from entirely different angles.
# in summary NOTE: if we initialize all 3 of I1's angles to be 0, and we don't care about reaching a particular orientation, then the choice of convention doesn't matter.
I1 = N.orientnew('I1', 'Body', [q1x, q1y, q1z], 'ZXZ' ) # may need to change to just "orient"
I2 = I1.orientnew('I2', 'Axis', [q2z, I1.z]) # changed w.r.t.

# Use Generalized speeds to set Angular velocities of Ref Frames
I1.set_ang_vel(N, u1x * N.x + u1y * N.y + u1z * N.z) # may have to aline somehow with chosen 'ZXZ'
I2.set_ang_vel(I1, u2z * I1.z)

## Define and set locations for the points:
# The hinge
O = Point('O')
# 2 Centers of Mass
p1 = O.locatenew('p1', s/2 * I1.x + s/2 * I1.y) # NOTE centers are defined w.r.t. the corners, as O is put at the corner attachment between cubes 1 and 2. Note also they are defined in z plane where z = 0.
p2 = O.locatenew('p2', s/2 * I2.x + s/2 * I2.y) # NOTE later could change this w.r.t. p1, to suit trend for many cubes, but for 2 cubes it's easier to track O in graphs this way.
# NOTE NOTE NOTE NOTE READ ME!!!!!!!!!!!!!!!!!!!!!!!!!!! In augmenting to 8 cubes and defining points it would be far easier to think of initial configuration as the line, and just making sure to place the coils at the correct hinges. To start as a 8-sat cube in the simulation, you then just have to change the initial states and set points.

# kinematical differential equations (i.e. enforce qd=u)
kde = [qxd - ux, qyd - uy, qzd - uz, q1xd - u1x, q1yd - u1y, q1zd - u1z, q2zd - u2z] # each of these =0, i.e. qxd=ux

## Set Linear Velocities
# Double Pendulum sets O's velocity to 0, i.e.  O.set_vel(N, 0), but we want O to be able to move w.r.t. N as it's in space, and which is why have generalized co-ordinates for exactly that- set equal to those velocities.
O.set_vel(N, ux*N.x + uy*N.y + uz*N.z)


# Set linear velocities of the Points using v2=v1+Omega-cross-r_21 ; This is a required step that's needed to explicitly set the linear velocities of the points in the inertial frame.
# For points p_i and O fixed in a frame (I), that is rotating in frame N, p_i.v2pt_theory(O, N, I) gives p_i's velocity in frame N (as it's fixed in I, and was defined in terms of I)
p1.v2pt_theory(O, N, I1)
p2.v2pt_theory(O, N, I2)


## Define coil centers
# Coil Centers will define inter-coil distances that forces will be calculated from. Coils assumed to be 4mm radius, centered 50*4/10=20mm from cube's middle, i.e. with coil center 5mm from edge and coil perimeter 1mm from edge. We use the excel force equation found after 22/02/17 that calculates forces in "Y" direction using the coil centers: NOTE the distance to force model is perfectly valid down to 2mm closest separatation (10mm between centers); smaller distances isn't permitted physicaly by the real structure, so should not be permitted from occuring in simulation either.

## automated coil generation AND v2pt assignment for all 12 coils in cube 1 and 2
# All are defined with respect to that cube's CoM point (see CW convention, 16th Feb in notebook)

# set midpoint of coil w.r.t. cube middle; used by new way, and also by cube i on old way.
R = s*4/10
num_coils = 12
coeffs = zeros((num_coils,3))
coeffs[0,:]  = [-1, 1, 0]
coeffs[1,:]  = [ 1, 1, 0]
coeffs[2,:]  = [ 1,-1, 0]
coeffs[3,:]  = [-1,-1, 0]

coeffs[4,:]  = [ 0, 1,-1]
coeffs[5,:]  = [ 0, 1, 1]
coeffs[6,:]  = [ 0,-1, 1]
coeffs[7,:]  = [ 0,-1,-1]

coeffs[8,:]  = [-1, 0,-1]
coeffs[9,:]  = [-1, 0, 1]
coeffs[10,:] = [ 1, 0, 1]
coeffs[11,:] = [ 1, 0,-1]

c1 = []
c2 = []

for i in range(num_coils):
    coil_1_i = p1.locatenew('c1_'+ str(i+1), coeffs[i,0]*R * I1.x + coeffs[i,1]*R * I1.y + coeffs[i,2]*R * I1.z )
    coil_1_i.v2pt_theory(O, N, I1)
    c1.append(coil_1_i)

    coil_2_i = p2.locatenew('c2_'+ str(i+1), coeffs[i,0]*R * I2.x + coeffs[i,1]*R * I2.y + coeffs[i,2]*R * I2.z )
    coil_2_i.v2pt_theory(O, N, I2)
    c2.append(coil_2_i)


# Give the Origin of the Inertial frame w.r.t. O. A kind of simple hack as we need this as a Point for the simulation, and there seems to be no clean way of accessing the inertial frame's origin.
N_o = Point('N_o')
N_o.set_pos(O, -qx*N.x -qy*N.y - qz*N.z)

## Rigid Body definitions
# Double Pendulum uses just point masses in BL (body list) that accompanies FL (force list) as an argument in the kanes_equations method. We follow the Human Standing tutorial, where rigid bodies are defined i.t.o. both mass and inertia

# Set masses (constant values)
mass_1, mass_2 = symbols('mass_1, mass_2')
# Set inertias (constant values). We are modelling a 2D planar problem so only need inertia in Z.
inertia_1, inertia_2 = symbols('I_1, I_2')
# use inertia() function for convenience to create inertia dyadic (basis dependent tensor)
inertia_dyadic_1 = inertia(I1,inertia_1,inertia_1,inertia_1)
inertia_dyadic_2 = inertia(I2,inertia_2,inertia_2,inertia_2)
# Also the to_matrix() method, e.g. inertia_dyadic_j.to_matrix(J), converts dyadic to a matrix expressed in a specified frame

# Set point to which inertias are defined with respect to. Choices should be to define w.r.t. CoM, or to define w.r.t. O and use parallel Axis Theorem. We will set w.r.t. CoM as that is done in tutorial.
inertia_center_1 = (inertia_dyadic_1 , p1)
inertia_center_2 = (inertia_dyadic_2 , p2)

# Define rigid body in terms of mass center, reference frame, mass, and inertia defined about a given point
body_1 = RigidBody('body_1', p1, I1, mass_1, inertia_center_1)
body_2 = RigidBody('body_2', p2, I2, mass_2, inertia_center_2)


### Forces


## Compute the magnitude of the force vector using EM equation
def compute_EM_Force(dist,IiIj):
    """
    Force = F(dist,amp1,amp2)
    Returns force calculated from EM script, and fitted to curve in excel
    """
    # The force equation below is F/IiIj, as it was determined using both currents=1A, which is why product IiIj is multiplied in. IiIj is determined from the controller at each time step and may be either pos or neg, while Pull pair regulates speed of approach. Note the equation is simple because it is force "in y" between longitudinally aligned EMs, so it is approximated by simple 1d equation assuming perfect alignment.
    # Possible TODO perhaps improve the distance-to-force model. This force should be precise, using a more complex and well-fitted equation. It is the equation used by the controller, if it uses the dynamics, that may need simplification.

    ans = IiIj * 3 * 10**(-9) * dist**(-2.299) # Perfect fit for >2mm closest separation, i.e. 10mm between coil centers
    return ans



def appendLoads(loads,cube_i,cube_j,push_i,push_j,pull_i,pull_j , pushCur_ij , pullCur_ij):

    """ returns 4 forces that are appended to loads. i is the cube closer to the base cube in the kinematic chain. """
    # Possible TODO: account for the smaller cross-coupled forces that are turned on during control

    # cube_i and cube_j are cubes e.g. c1, c2, which hold a list of all 12 points of their coils
    # push_i,push_j,pull_i,pull_j are 4 coil locations
    # we pass in e.g. c1 which has list of all coils in the cube, then pass in index+1 separately for clarity.
    # function uses e.g. cube_i[push_i-1], as c5[0] is coil 1 on cube 5
    # pushCur_12 , pullCur_12 are the product IiIj for the coil pair, fonud from controller

    PushDistVec_ij = cube_j[push_j - 1].pos_from(cube_i[push_i - 1]).express(N)
    PushDistVec_ij_norm = PushDistVec_ij.normalize()

    PullDistVec_ij = cube_j[pull_j - 1].pos_from(cube_i[pull_i - 1]).express(N)
    PullDistVec_ij_norm = PullDistVec_ij.normalize()

    # Without loss of generality, we only need one force magnitude for each coil pair, as this depends on magnitude of distance and product of their currents.
    PushSize = compute_EM_Force(PushDistVec_ij.magnitude() , pushCur_ij )
    PullSize = compute_EM_Force(PullDistVec_ij.magnitude() , pullCur_ij )
    # two forces per coil pair are needed to capture the opposite direction, and these are sent to the dynamics. They encode both Force magnitude and vector direction.
    PushVec_ij = PushSize * PushDistVec_ij_norm
    PushVec_ji = PushSize * (-1 * PushDistVec_ij_norm)
    PullVec_ij = PullSize * PullDistVec_ij_norm
    PullVec_ji = PullSize * (-1 * PullDistVec_ij_norm)

    ## Create tuple holding the vector, and the point upon which it acts
    PushForce_ji = (cube_i[push_i - 1] , PushVec_ji)
    PushForce_ij = (cube_j[push_j - 1] , PushVec_ij)
    PullForce_ji = (cube_i[pull_i - 1] , PullVec_ji)
    PullForce_ij = (cube_j[pull_j - 1] , PullVec_ij)

    loads.append(PushForce_ji)
    loads.append(PushForce_ij)
    loads.append(PullForce_ji)
    loads.append(PullForce_ij)

    return loads

# Push Current IiIj and Pull Current IiIj are scalar currents. These are later put in specified, and then at every time step, their values are calculated by controller, sent here to compute_EM_Force and create Force vectors.
pushCur_12, pullCur_12 = dynamicsymbols('pushCur_12, pullCur_12')
# for cubes 2 to 3 we will add: pushCur_23, pullCur_23


loads = []
loads = appendLoads(loads,c1,c2,3,1,1,3 , pushCur_12 , pullCur_12)




## Use Kane's Method to generate Equations of Motion
# Put Generalized coordinates in vector
coordinates = [qx, qy, qz, q1x, q1y, q1z, q2z]
# Put Generalized speeds in vector
speeds = [ux, uy, uz, u1x, u1y, u1z, u2z]
# Create KanesMethod object, and vectors holding forces and rigid bodies
kane = KanesMethod(N, coordinates, speeds, kde)

bodies = [body_1, body_2]

# Generate Fr and Fr_star that are used by Kane
fr, frstar = kane.kanes_equations(loads, bodies)

mass_matrix = kane.mass_matrix_full
forcing_vector = kane.forcing_full

### Transform symbolic equations of motion to Python functions and evaluate using numerical integration to solve the ordinary differential initial value problem

# List all constants used by the EoM: lengths, angles, masses, inertias (i.e. anything defined as "symbols"). Order doesn't matter.
constants = [s, mass_1, mass_2, inertia_1, inertia_2]

# NOTE CHANGES. A list called "specified" holds symbols representing values that can be changed ("specified") during run time, e.g. be dependent on state variables. This is where the control happens.
specified = [pushCur_12, pullCur_12]


# create the function
right_hand_side = generate_ode_function(forcing_vector, coordinates, speeds, constants, mass_matrix = mass_matrix, specifieds=specified)

## Set initial conditions, parameter values and time array
# Assign numerical values to all constants (and any exogenous inputs)
numerical_constants = array([
                            0.05, # side length 50mm
                            0.25, # 250g
                            0.25, # 250g
                            (0.25*0.05**2)/6, # I of CoM of cube= (ms^2)/6
                            (0.25*0.05**2)/6 # I of CoM of cube= (ms^2)/6
                            ])

###################
# Computing the derivates and integrating
###################

# First: right_hand_side
# right_hand_side is the symbolic ODE function that is generated (using a function called "generate_ode_function") from all the kinematic stuff and kanes equations. It takes as arguments the initial conditions (4 positions q and 4 velocities u), the current time, additional functions/symbols/forces, are lastly values for numerical constants. When you pass these arguments (encoding forces, kinematic constraints etc.) to it, it generates the derivates of the state vector [p,q] and gives you actual numbers for these (not more symbols), i.e. numbers for velocities and accererations, given the current state and force.

# Second: integration step.
# Old method: Odeint - this takes as arguments the right_hand_side function, followed by all the same arguments used to initialize right_hand_side. Odeint is nothing special, it basically multiples the derivates with time, adds result to previous state, and gives new states. The problem is, you give it the ICs, dynamics and forces and it spits out all the state variables for all time at the very end. But, we can't seem to access the actual iteration index, which would be nice in order to be able to change setpoints/forces at given indexes of time. Also quite annoying that you cant evaluate values for the states e.g. to use as conditionals.
# Current method: we manually integrate right_hand_side one time step at a time, i.e. call an integration function iteratively, and save the variables as we go. This just allows more control by having direct access to the iteration index "i", and thereby t, and all the state variables at a given time step, and call also update/change control in time.



## Initialize q and u in xo
# Angle states: Note CCW is +, and 0 is East.
num_states = 14 # qx, qy, qz, q1x, q1y, q1z, q2z + derivatives
x0 = zeros(num_states)
x0[5] = deg2rad(0.0) # cube 1, angle in z
x0[6] = deg2rad(-90.0) #deg2rad(-270.0)  # cube 2, angle in z, angle defined w.r.t. cube 1's frame, and defined w.r.t. same hinge (O), so must be defined at -90 away (i.e. 90 CW), otherwise they're in the same place. This means the 1-2 cube pair will use different coils for a given rotation, compared to the additional cubes, as new hinge points will be made and each cube defined w.r.t. the preceding point, and so each of their angle states can be initialized at 0 degrees w.r.t. previous cube's frame.

## Setpoints
# Turning clockwise around a corner (as opposed to traversals, which we don't do in this script) means turning 180 CW (i.e. -180) from previous cube.
setPoint = zeros(numCubes - 1) # no setpoint for 1st cube w.r.t inertial
setPoint[0] = deg2rad(-270.0) # will probably be either 270 or -270 for all, if all states initialized as 90 or -90 from previous.

# Time vector and sampling frequency
T_end = 80#240
Hz = 60
inc = Hz * T_end + 1
t = linspace(0,T_end,inc)

# Initialize errors, inputs and debugging variables
e = zeros(numCubes - 1) # Error. Same number of errors as setpoints
e_prev = zeros(numCubes - 1) # will hold previous error.
e_sum = zeros(numCubes - 1) # will hold sum of n previous errors, for Ki
e_mat = zeros((len(t),numCubes - 1)) # holds Errors for all time
numSpec_mat = zeros((len(t),1)) # TODO expand columns for 8 cubes.
U = zeros(numCubes - 1) # Input. Same number of inputs as setpoints and errors
dedt = zeros(numCubes - 1)
e_debug_act = zeros(numCubes - 1) # debugging
e_debug_act_mat = zeros((len(t),numCubes - 1)) # debugging
dedt_mat = zeros((len(t),numCubes - 1)) # debugging
sig_initPivot = zeros(numCubes - 1) # will set from 0 (False) to 1/True at time pivot is desired
t_initPivot = zeros(numCubes - 1) # time at which sig_initPivot will be switched
t_initPivot[0] = 10

# Integration parameters
dt = t[1]-t[0]
y = zeros((len(t),num_states))
y[0,:] = x0
dydt_vec = zeros((len(t),num_states))

def controller(state,e_prev,e_sum, setPointC):

    # The controller will pass input values to inputsToCurrents(), which will in turn set numerical_specified, which are the currents put into the force equation which gives forces that are used to generate the derivatives at every time step.
    # NOTE it appears that some Initial conditions can't be stabilized with "feasible" currents, e.g. setting initial speeds to 30 deg/s. In particular, after briefly crossing the setpoint and appearing semi-stable, when angles crosses back over to negative, it quickly diverges. The same can happen after a cross over in the opposite direction. However, when initializing slightly off setpoint with speed in wrong direction, it can go back to setpoint. The former problem is remedied by very big Pull currents; These observations are likely the result of 2 things: that when you give theta 2 a large initial velocity, due to the center of mass of the whole system, the cubes "want" to flay out, so increasingly difficult to stay together, and in addition, the force's and thus controller's weakness at larger errors/distances
    # NOTE control U set to negative for positive real errors. Don't know why this is required to work, as the implementation both in control section and in sympy looks correct.
    # NOTE regardless, cubes should not be permitted to go past the setpoint, not only because the control can go funny, but mainly because it's a physical violation: This also means you don't need to worry about this for hardware.

    SP = setPointC


    e_debug_act = rad2deg(SP - state)
    e = abs(SP - state) # Note the absolute.
    e = e%(2*pi)
    # NOTE e[0] is always >0, so no complication to modulus operator from negatives. Note also in our setup, errors should never be more than 180 deg (a traversal = 180, pivot = 90), so dont need to worry about converting 181 deg to 179 to go "the other way", as the other way would be physically blocked anyway.

    # TODO note that states[3] is theta2: first state under control in 2-cube scenario. For 8 cubes, want a loop to iterate over each cube.
    dedt = (e - e_prev) / dt
    # if e was not absolute, and we are going from a neg value toward a more negative setpoint, then dedt would be positive.
    Kp = -10#-10#-0.01
    Kd = -80#-40#-100 # Kd>Kp ensures no overshoot
    Ki = -0.3
    U = Kp * e + Kd * dedt + Ki * e_sum
    # if absolute error (always pos) is big, i.e. error in any direction, we want to pull (negative force), and if dedt is negative, i.e. we are approaching set point, then we want to multiply that with negative to push it.

    ######## NOTE This HACK is working. It seems like the code is doing exactly what you want, but it doesn't have the desired result
    if (e_debug_act >= 0):
        U = -U

    # if dist is very small, set U=0 to avoid possible explosions in force
    if (e < deg2rad(0.01)):
        U = 0

    return U , e , e_debug_act, dedt

def inputsToCurrents(inputs,signalPivot):

    # NOTE:
    # U is to be put in to Ii and Ij
    # Physical meaning:
    # if U is positive, coils should have same current direction
    # if U is negative, coils should have opposite current directions
    # Recall, all that matters is Ii*Ij (magnitude and sign of product) for one pair of coils. If decide to distribute U equally between Ii and Ij, be careful of sqrting neg U.

    #TODO if signal pivot is true, exectute the below. Else, stay.

    U = inputs
    amps = zeros(2)
    # Maximums are important A) because we must be realistic with physical power available, and also to mitigate explosions in force equation at low distances.
    push_max = 0.8
    pull_max = 1.5 # ensure pull_max > push_max ; see notes

    push_1_and_2 = push_max
    if (abs(U) < pull_max):
        pull_1_and_2 = U
    else:
        pull_1_and_2 = sign(U) * pull_max

    if (signalPivot==True):
        amps[0] = push_1_and_2 # push from starting config
        amps[1] = pull_1_and_2 # pull toward target config
    else: # if haven't triggered the pivot yet
        amps[0] = pull_1_and_2 # pull toward initialized state.
        amps[1] = 0 # do nothing with the coil pair at the final goal state.

    return amps



def setErrorPrevAndSum(i,state,setPointC):

    SP = setPointC
    # set previous error to use in controller by Kd
    if (i==0):
        e_prev = abs(SP - state) # same as e will be on i=0
        e_prev = e_prev%(2*pi)
    else:
        e_prev = e_mat[i-1,:]

    # set error sum to use in controller by Ki
    if (i>25):
        e_sum = np.sum(e_mat[i-15:i , 0], axis = 0)
    else:
        e_sum = 0

    return e_prev,e_sum



def appendNumSpec(numspec_current,numspec_toAdd):
    numspec_current.append(numspec_toAdd)
    return numspec_current



## Integration
for i in range(len(t)):

    states = y[i,:]
    time = t[i]

    # numerical_specified_tmp is calculated by inputstocurrents for each cube in a loop, and then appended to numerical_specified; a similar procedure as done with forces.
    numerical_specified = []

    for C in range(numCubes-1):
        if (time >= t_initPivot[C]):
            sig_initPivot[C] = True
            SP = setPoint[C] # assign the passed setpoint "SP" to globally defined one
        else:
            sig_initPivot[C] = False
            SP = x0[C+6] # override setpoint, setting to initialized state.
        state = states[C+6] # For clarity; +6 as the 7th state is the first controlled one.
        e_prev[C],e_sum[C] = setErrorPrevAndSum(i,state,SP)
        inputs, e[C], e_debug_act[C], dedt[C] = controller(state,e_prev[C],e_sum[C],SP)
        numerical_specified_tmp = inputsToCurrents(inputs,sig_initPivot[C])
        # append each value in numerical_specified_tmp individually, otherwise you get a list in an array, and right_hand_side doesn't like that.
        for num in range(len(numerical_specified_tmp)):
            numerical_specified.append(numerical_specified_tmp[num])
    # input is not indexed, it is overwritten on each C loop as it's just passed to inputstocurrents which makes numspec, and numspec is appended on every loop to make one Current list for all cubes, as just one list is passed to RHS.

    dydt = right_hand_side(y[i,:], t[i], numerical_specified, numerical_constants)
    dydt_vec[i,:] = dydt
    if (i == (len(t)-1)): # i.e. if on last iteration
        break
    y[i+1,:] = y[i,:] + dt * dydt
    e_mat[i,:] = e
    e_debug_act_mat[i,:] = e_debug_act
    dedt_mat[i,:] = dedt
    numSpec_mat[i,:] = numerical_specified[1]
    if (i%(Hz/4) == 0):
        print(i/Hz,'||',numerical_specified[1],'||',e_debug_act)


def plot():

    plt.close('all')
    # Plot x and y
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(t, y[:,0]*1000, label = 'x')
    ax1.plot(t, y[:,1]*1000, label = 'y')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Distance [mm]')
    ax1.legend()
    # Plot angles
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(t, rad2deg(y[:,5]), label = 'theta 1')
    ax2.plot(t, rad2deg(y[:,6]), label = 'theta 2')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Angle [deg]')
    ax2.legend()
    # plot velocities
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(t, rad2deg(y[:,12]), label = 'theta 1 dot')
    ax3.plot(t, rad2deg(y[:,13]), label = 'theta 2 dot')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Angular velocity [deg]')
    ax3.legend()
    # plot error
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(t, rad2deg(e_mat[:,0]), label = 'error')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Error [deg]')
    ax4.legend()
    # plot currents
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    ax5.plot(t, numSpec_mat[:,0], label = 'Current')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Current [A]')
    ax5.legend()
    # plot actual error, not absolute used in calcs
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111)
    ax6.plot(t, e_debug_act_mat[:,0], label = 'real signed error')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Real signed error [deg]')
    ax6.legend()
    # plot actual error, not absolute used in calcs
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(111)
    ax7.plot(t, dedt_mat[:,0], label = 'de/dt')
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('de/dt [deg/s]')
    ax7.legend()



### Visualization
def visualize():
    # Create shapes
    cube_1_shape = Cube(color='darkblue', length = 0.05)
    cube_2_shape = Cube(color='darkred', length = 0.05)

    # create frames to attach to shapes
    cube_1_frame = VisualizationFrame(I1, p1, cube_1_shape)
    cube_2_frame = VisualizationFrame(I2, p2, cube_2_shape)

    # create scene with a frame base and origin
    scene = Scene(N,N_o)
    # append frames that should be visualized
    scene.visualization_frames = [cube_1_frame,
                                  cube_2_frame]
    # provide constants
    constants_dict = dict(zip(constants, numerical_constants))

    # give symbolic states and their values
    scene.states_symbols = coordinates + speeds
    scene.constants = constants_dict
    scene.states_trajectories = y

    #visualize
    scene.frames_per_second=800 # default is 30, which is very slow
    scene.display()


# TODO plot graph showing forces at the hinge over time, with a second axis showing the current current required to maintain that force at that close range.



plot()
#visualize()
print(int(myTime.time() - start_time), 'seconds') # Print wall time
