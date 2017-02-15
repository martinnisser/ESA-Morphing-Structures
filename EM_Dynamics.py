# EM dynamics
# Script to generate and solve the equations of motion for two EM-driven EM satellites
from sympy import *
from sympy.physics.mechanics import *

## Generalized co-ordinates x, y, theta1/I, theta2/J
# q3, q4 are counter clockwise (right hand rule), x omponents giving the angles of Cube CoMs
q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
q1d, q2d, q3d, q4d = dynamicsymbols('q1 q2 q3 q4', 1)
# Derivatives of Generalized Co-ordinates
u1, u2, u3, u4 = dynamicsymbols('u1 u2 u3 u4')
u1d, u2d, u3d, u4d = dynamicsymbols('u1 u2 u3 u4', 1)
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
# TODO vectors should be changed to act at the EM coils, not mass CoMs. Will probably need to make extra points for this.

vec_ij = p_j.pos_from(p_i)
vec_ij_norm = vec_ij.normalize()

vec_ji = p_i.pos_from(p_j)
vec_ji_norm = vec_ji.normalize()

## Compute the magnitude of the force vector using EM equation

def compute_EM_Force(dist):
    #####
    # TODO implement correct model
    #####
    return 1/dist**2

vec_ij_mag = compute_EM_Force(vec_ij.magnitude())
vec_ji_mag = compute_EM_Force(vec_ji.magnitude())

## TODO set vector using magnitude and direction

# TODO Create tuple holding the vector, and the point upon which it acts

## TODO Will at some point have to make points/forces on catching side. 

# TODO also try to plot these things early on, to debug e.g. correct direction of angles and transformation matrices (it seems positive angles are CCW, so see e.g. red notes in notebook about swapping angles in your derivation)




## Other Usefuls to know about
# print(m_i.pos_from(m_j)) # position of m_i from m_j
# print(N.dcm(I)) #DCM: format is N.xyz = N.dcm(I) * I.xyz
