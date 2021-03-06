# EM dynamics
# Script to generate and solve the equations of motion for two EM-driven EM satellites
# For help see PyDy Tutorials: https://github.com/pydy/pydy-tutorial-human-standing http://www.pydy.org/examples/double_pendulum.html


from sympy import *
from sympy.physics.mechanics import *
from numpy import deg2rad, rad2deg, array, zeros, linspace
from scipy.integrate import odeint
from pydy.codegen.ode_function_generators import generate_ode_function
from matplotlib.pyplot import plot, legend, xlabel, ylabel, rcParams
rcParams['figure.figsize'] = (14.0, 6.0)

## Generalized co-ordinates x, y, theta1/I, theta2/J
# q3, q4 are counter clockwise (right hand rule), x omponents giving the angles of Cube CoMs
q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
q1d, q2d, q3d, q4d = dynamicsymbols('q1 q2 q3 q4', 1) # skipped in human tut.
# Derivatives of Generalized Co-ordinates
u1, u2, u3, u4 = dynamicsymbols('u1 u2 u3 u4')
u1d, u2d, u3d, u4d = dynamicsymbols('u1 u2 u3 u4', 1) # skipped in human tut.
# Parameters to set
r, R, phi1, phi2 = symbols('r, R, phi1, phi2')

# Reference Frames
N = ReferenceFrame('N') # Inertial frame
I = N.orientnew('I', 'Axis', [q3, N.z])
J = N.orientnew('J', 'Axis', [q4, N.z])

# Use Generalized speeds to set Angular velocities of Ref Frames
I.set_ang_vel(N, u3 * N.z)
J.set_ang_vel(N, u4 * N.z)

# Define and set locations for 3 points: The hinge and 2 Centers of Mass
O = Point('O')
p_i = O.locatenew('p_i', r * I.x)
p_j = O.locatenew('p_j', r * J.x)

# kinematical differential equations (i.e. enforce qd=u)
kde = [q1d - u1, q2d - u2, q3d - u3, q4d - u4] # each of these =0, i.e. q1d=u1

## Set Linear Velocities
# Double Pendulum sets O's velocoty to 0, i.e.  O.set_vel(N, 0), but we want O to be able to move w.r.t. N as it's in space, and which is why have generalized co-ordinates for exactly that- set equal to those velocities.
O.set_vel(N, u1*N.x + u2*N.y)
# Set linear velocities of the Points using v2=v1+Omega-cross-r_21
# For points p_i and O fixed in a frame (I), that is rotating in frame N, p_i.v2pt_theory(O, N, I) gives p_i's velocity in frame N (as it's fixed in I)
p_i.v2pt_theory(O, N, I)
p_j.v2pt_theory(O, N, J)

## Rigid Body definitions
# Double Pendulum uses just point masses in BL (body list) that accompanies FL (force list) as an argument in the kanes_equations method. We need to follow the Human Standing tutorial, where rigid bodies are defined i.t.o. both mass and inertia

# Set masses (constant values)
mass_i, mass_j = symbols('m_i, m_j')
# Set inertias (constant values). We are modelling a 2D planar problem so only need inertia in Z.
inertia_i, inertia_j = symbols('I_i, I_j')
# use inertia() function for convenience to create inertia dyadic (basis dependent tensor)
inertia_dyadic_i = inertia(I,0,0,inertia_i)
inertia_dyadic_j = inertia(J,0,0,inertia_j)
# Also the to_matrix() method, e.g. inertia_dyadic_j.to_matrix(J), converts dyadic to a matrix expressed in a specified frame

# Set point to which inertias are defined with respect to. Choices should be to define w.r.t. CoM, or to define w.r.t. O and use parallel Axis Theorem. We will set w.r.t. CoM as that is done in tutorial.
inertia_center_i = (inertia_dyadic_i , p_i)
inertia_center_j = (inertia_dyadic_j , p_j)

# Define rigid body in terms of mass center, reference frame, mass, and inertia defined about a given point
body_i = RigidBody('body i', p_i, I, mass_i, inertia_center_i)
body_j = RigidBody('body j', p_j, J, mass_j, inertia_center_j)


### Forces

## Create vector direction through which forces (both F12 and F21) act.
# TODO vectors should be changed to act at the EM coils, not mass CoMs. The way to do this seems to be to extra points that arefixed in each body frame.
# TODO These should be explicitly expressed in inertial frame right? Tutorial expresses Gravity in inertial frame. Confirm this is having desired result.
vec_ij = p_j.pos_from(p_i).express(N)
vec_ij_norm = vec_ij.normalize()

vec_ji = p_i.pos_from(p_j).express(N)
vec_ji_norm = vec_ji.normalize()


## Compute the magnitude of the force vector using EM equation
def compute_EM_Force(dist):
    #####
    # TODO Augment with control of polarization, and perhaps improve the distance-to-force model.
    #####
    return 0.0003*dist**(-0.944)
Force_vec_ij_mag = compute_EM_Force(vec_ij.magnitude())
Force_vec_ji_mag = compute_EM_Force(vec_ji.magnitude())


## Set vector using Force magnitude and vector direction
Force_vec_ij = Force_vec_ij_mag * vec_ij_norm
Force_vec_ji = Force_vec_ji_mag * vec_ji_norm


## Create tuple holding the vector, and the point upon which it acts
# TODO update with correct EM point, not CoM point
# NOTE reversal of indices here: Force_ji is consistent with F21 from EM.py
# Force_vec_ij acts on p_j to give Force_ji.  Make sure to put in correct order when Forces are put into matrices and solved row by row in ODE integrator
Force_ji = (p_j , Force_vec_ij)
Force_ij = (p_i , Force_vec_ji)


## Exogeneous Inputs (forces and torques)- will be set to 0 later.
x_F, y_F, i_T, j_T = dynamicsymbols('x_F, y_F, i_T, j_T')
# As done with F_ij and F_ji above, need to multiply magntidue by directions.
Ex_F_vector = x_F * N.x + y_F * N.y
# NOTE Not sure about Ex_T eqs, but will be setting mag to 0 anyway
Ex_T_vector_theta_i = i_T * N.z - j_T * N.z
Ex_T_vector_theta_j = j_T * N.z - i_T * N.z
# And create tuple of point, force or frame, torque
Ex_Force = (O, Ex_F_vector)
Ex_Torque_i = (I, Ex_T_vector_theta_i)
Ex_Torque_j = (J, Ex_T_vector_theta_j)



## Use Kane's Method to generate Equations of Motion
# Put Generalized coordinates in vector
coordinates = [q1, q2, q3, q4]
# Put Generalized speeds in vector
speeds = [u1, u2, u3, u4]
# Create KanesMethod object, and vectors holding forces and rigid bodies
kane = KanesMethod(N, coordinates, speeds, kde)
loads = [Force_ij, Force_ji, Ex_Force, Ex_Torque_i, Ex_Torque_j]
bodies = [body_i, body_j]
# Generate Fr and Fr_star that are used by Kane
fr, frstar = kane.kanes_equations(loads, bodies)
# Reduce expression by using known trig identities
trigsimp(fr + frstar)
# Simplify the Mass matrix and Force vector
mass_matrix = trigsimp(kane.mass_matrix_full)
forcing_vector = trigsimp(kane.forcing_full)


### Transform symbolic equations of motion to Python functions and evaluate using numerical integration to solve the ordinary differential initial value problem

# List all constants used by the EoM: lengths, angles, masses, inertias (i.e. anything defined as "symbols"). Order doesn't matter.
constants = [r, mass_i, mass_j, inertia_i, inertia_j] # NOTE add in R, phi1, phi2 when make extra points.

# A list called "specified" holds Exogenous inputs, i.e. inputs that don't rely on information of system (i.e. externally applied forces or torques). The Tutorial includes them, and defined them after Forces, including them in "loads" list above. I tried to neglect these and change remaining code accordingly, but without success. Instead we just create external loads as done in tutorial and set them to 0 for all time.

specified = [x_F, y_F, i_T, j_T]

# create the function
right_hand_side = generate_ode_function(forcing_vector, coordinates, speeds, constants, mass_matrix = mass_matrix, specifieds=specified)

## Set initial conditions, parameter values and time array

# Initialize q and u
# righthandside function creates state vector x holding q and u
x0 = zeros(8)
x0[2] = deg2rad(315.0) # i.e. -pi/4 , assuming CCW+, so will be right cube.
x0[3] = deg2rad(45.0)  # i.e. +pi/4 , assuming CCW+, so will be left cube.

# Assign numerical values to all constants (and any exogenous inputs)
numerical_constants = array([(2*0.025**2)**0.5, # Hypot. of side 25mm
                            0.25, # 250g
                            0.25, # 250g
                            (0.25*0.05**2)/6, # I of CoM of cube= (ms^2)/6
                            (0.25*0.05**2)/6 # I of CoM of cube= (ms^2)/6
                            ])

# numerical_specified created here and set to 0
numerical_specified = zeros(4)

# create time vector
t = linspace(0,10,601) # 60 Hz

# Evaluate right_hand_side numerically
# This seems to basically intitialize the equation: The 0.0 here is the current time, and x) initilaizes the states. Seems to require numerical_specified as an argument before numerical_constants.
right_hand_side(x0, 0.0, numerical_specified, numerical_constants)

# integration step
y = odeint(right_hand_side, x0, t, args=(numerical_specified, numerical_constants))
print(y.shape)

### Plotting
plot(t, rad2deg(y[:, 2:4]))
xlabel('Time [s]')
ylabel('Angle [deg]')
#legend()



# TODO Create points and forces on catching side.
# TODO Control theory - NOTE the same Human Tutorial includes a Control part
# TODO In plots, check in particular to ensure correct direction of angles and transformation matrices (it seems positive angles are CCW, so see e.g. red notes in notebook about swapping angles in your derivation)

# print(N.dcm(I)) #DCM: format is N.xyz = N.dcm(I) * I.xyz
