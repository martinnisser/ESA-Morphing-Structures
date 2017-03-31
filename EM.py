# EM
# Created by Martin Nisser, 16th January 2017
# Script to calculate EM forces and torques between 2 arbitrarily oriented coils, or two parallel wires

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
plt.ion() # interactive plotting on

## Global Variables.
# Physical Constants
MuO= 4*np.pi*10**(-7)
pi = np.pi
# Start Clock
t0 = time.clock()
# Set Wire Geometry to output of chosen Function
ChooseCoils = True
ChooseParallelLines = not ChooseCoils
# Discretization (N)
Ni = 300#! 3000
Nj = Ni
# Currents (I)
Ii = 1
Ij = Ii
## Parallel wires geometry
X_length = 200
## Coils Geometry
# Coil i
turns_i = 10#! 200
radius_i = 4 / 1000 # enter radius in mm to left of "/"
h_i = 0.1 / (2*pi*1000)#! 0.2 / (2*pi*1000) # enter height of one turn of coil in mm to the left of "/"
# Coil j
turns_j = turns_i
radius_j = radius_i
h_j = h_i
# check radius/height is ok for coils
if (radius_i/(h_i*(2*pi)) < 10) or (radius_j/(h_j*(2*pi)) < 10) :
    print('\n\nWarning, coil pitch is steep: r/h < 10. ri/hi=',radius_i/(h_i*(2*pi)))

def main():
    d = np.array([0,0,100]) / 1000  #! d = np.array([0,0,40.2]) / 1000 # separation along axis between coils in mm
    #iterateCoils(d)
    oneCoilOrLines(d)

def iterateCoils(distance):

    d=distance
    N_iter = 40 #! 80
    F12_results = np.zeros((N_iter,3))
    F21_results = np.zeros((N_iter,3))
    T12_results = np.zeros((N_iter,3))
    T21_results = np.zeros((N_iter,3))
    time_comp_results = np.zeros(N_iter)
    y_sep_results = np.zeros(N_iter)
    z_sep_results = np.zeros(N_iter)
    d_results = np.zeros((N_iter,3))
    # Coil separation
    increment= np.array([0,0,0.01]) / 1000 #! increment= np.array([0,0,0.2]) / 1000

    print('start time: ',time.strftime("%H:%M:%S"))
    print('Iterations: ',N_iter)
    for n in range(N_iter):
        #compute
        result = ComputeForcesAndTorques(d)
        F12_results[n,:]= result[0]
        F21_results[n,:]= result[1]
        T12_results[n,:]= result[2]
        T21_results[n,:]= result[3]
        time_comp_results[n]= result[4]
        y_sep_results[n]= result[5]
        z_sep_results[n]= result[6]
        d_results[n,:] = d*1000
        d= np.sum([d,increment],axis=0)
        print(n)
    print(F12_results,'\n',d_results)
    np.savetxt("F12_results.csv", F12_results, delimiter=",")
    np.savetxt("F21_results.csv", F21_results, delimiter=",")
    np.savetxt("T12_results.csv", T12_results, delimiter=",")
    np.savetxt("T21_results.csv", T21_results, delimiter=",")
    np.savetxt("time_comp_results.csv", time_comp_results, delimiter=",")
    np.savetxt("y_sep_results.csv", y_sep_results, delimiter=",")
    np.savetxt("z_sep_results.csv", z_sep_results, delimiter=",")
    np.savetxt("d_results.csv", d_results, delimiter=",")

def oneCoilOrLines(distance):
    #compute
    print('start time: ',time.strftime("%H:%M:%S"))
    d= distance
    ComputeForcesAndTorques(d)

## Far Field model: Force of one dipole on another, axially aligned
def FarFieldModel(distance):
    d=distance
    r_F = radius_i
    d_F = d[2] # distance between two dipoles, taking distance in z from near-field
    N_F = turns_i
    I_F = Ii # Amps
    A_F = pi * r_F**2
    MuA_F= N_F*I_F*A_F # rough approximation?
    MuB_F= MuA_F # identical coils
    Force_F = 3*MuO*MuA_F*MuB_F/(2*pi*d_F**4) # equation for axially algined dipoles
    print('\n\nFar Field Model: Force =',Force_F)


def MakeParallelLines():
    # Set Wire Geometry to two parallel wires, spaced 1m apart
    dLi_Nodes = np.array([np.linspace(0,X_length,Ni+1),np.zeros(Ni+1),np.zeros(Ni+1)])
    dLj_Nodes = np.array([np.linspace(0,X_length,Nj+1),np.ones(Nj+1),np.zeros(Nj+1)])
    return dLi_Nodes,dLj_Nodes

def MakeCoils(distance):

    d=distance
    # Parametrize coils
    theta_i = np.linspace(0,turns_i * 2*pi,Ni+1)
    x_i = radius_i * np.cos(theta_i)
    y_i = radius_i * np.sin(theta_i)
    z_i = h_i * theta_i # height of one turn is h_i

    # forces in Z are (virtually) cancelled using negative on np.sin(theta_j) which makes them symmtric in y. Currently forces in z are not 0 as coils are not symmetric, and force is prop to d^4. Plot both cases and inspect for clarity.
    theta_j = np.linspace(0,turns_j * 2*pi,Nj+1)
    x_j = radius_j * np.cos(theta_j) + d[0]
    y_j = radius_j * np.sin(theta_j) + d[1]
    z_j = h_j * theta_j + d[2]

    dLi_Nodes = np.array([x_i,y_i,z_i])
    dLj_Nodes = np.array([x_j,y_j,z_j])
    return dLi_Nodes,dLj_Nodes


def PlotWires(dLi_Nodes,dLj_Nodes,dLi_Middles,dLj_Middles):
    plt.close()
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Convert m to mm
    dLi_Nodes = dLi_Nodes*1000
    dLj_Nodes = dLj_Nodes*1000
    dLi_Middles = dLi_Middles*1000
    dLj_Middles = dLj_Middles*1000

    ax.plot(dLi_Nodes[0,:],dLi_Nodes[1,:],dLi_Nodes[2,:], c='b')
    ax.plot(dLj_Nodes[0,:],dLj_Nodes[1,:],dLj_Nodes[2,:], c='g')

    ax.scatter(dLi_Nodes[0,:],dLi_Nodes[1,:],dLi_Nodes[2,:], c='b', depthshade=True)
    ax.scatter(dLj_Nodes[0,:],dLj_Nodes[1,:],dLj_Nodes[2,:], c='g', depthshade=True)

    ax.scatter(dLi_Middles[0,:],dLi_Middles[1,:],dLi_Middles[2,:], c='m', depthshade=True, marker = '+')
    ax.scatter(dLj_Middles[0,:],dLj_Middles[1,:],dLj_Middles[2,:], c='m', depthshade=True, marker = '+')


    '''
    # To also plot every vector connecting middles in i and j. Beware size! use N=10, 1 turn
    for x in range (Ni): # x iterates over i
        for y in range (Nj): # y iterates over j
            tmp_i = (dLi_Middles[0,x],dLi_Middles[1,x],dLi_Middles[2,x])
            tmp_j = (dLj_Middles[0,y],dLj_Middles[1,y],dLj_Middles[2,y])
            tmp= np.array([tmp_i,tmp_j])
            ax.plot(tmp[:,0], tmp[:,1],tmp[:,2],c='m')
    '''


    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    if (ChooseCoils==True):
        ax.set_xlim3d(-radius_i*1000*1.2, radius_i*1000*3.1)
        ax.set_ylim3d(-radius_i*1000*1.2, radius_i*1000*3.1)

    plt.show()

def ComputeForcesAndTorques(distance):

    d = distance

    if (ChooseParallelLines==True) and (ChooseCoils==False):
        dLi_Nodes = MakeParallelLines()[0]
        dLj_Nodes = MakeParallelLines()[1]
    elif (ChooseParallelLines==False) and (ChooseCoils==True):
        dLi_Nodes = MakeCoils(d)[0]
        dLj_Nodes = MakeCoils(d)[1]
    else:
        print('\nWARNING: You can only select one geometry to compute\n')


    # Initialize and compute dLi and dLj vectors and Middles, based on Nodes.
    dLi_Vectors = np.zeros((3,Ni))
    dLi_Middles = np.zeros((3,Ni))
    dLj_Vectors = np.zeros((3,Nj))
    dLj_Middles = np.zeros((3,Nj))

    for x in range(Ni):
        dLi_Middles[:,x]= (dLi_Nodes[:,x+1] + dLi_Nodes[:,x])/2
        dLi_Vectors[:,x]= dLi_Nodes[:,x+1] - dLi_Nodes[:,x]
    for x in range(Nj):
        dLj_Middles[:,x]= (dLj_Nodes[:,x+1] + dLj_Nodes[:,x])/2
        dLj_Vectors[:,x]= dLj_Nodes[:,x+1] - dLj_Nodes[:,x]

    # Compute Center of Mass (CoM) and R (the distance from CoM to each element's Middle)
    CoM_i = np.mean(dLi_Middles, axis=1)
    CoM_j = np.mean(dLj_Middles, axis=1)
    R_i = np.zeros((3,Ni))
    R_j = np.zeros((3,Nj))

    for x in range(Ni):
        R_i[:,x] = dLi_Middles[:,x] - CoM_i
    for x in range(Nj):
        R_j[:,x] = dLj_Middles[:,x] - CoM_j

    # Compute unit vectors of all di to all dLj, and vice versa
    r_ij= np.zeros((Ni,Ni,3))
    r_ji= np.zeros((Ni,Ni,3))
    r_ij_norm= np.zeros((Ni,Ni))
    r_ji_norm= np.zeros((Ni,Ni))
    r_ij_unit= np.zeros((Ni,Ni,3))
    r_ji_unit= np.zeros((Ni,Ni,3))

    for x in range (Ni): # x iterates over i
        for y in range (Nj): # y iterates over j
            r_ij[x,y,:] = dLj_Middles[:,y] - dLi_Middles[:,x]
            r_ji[y,x,:] = dLi_Middles[:,x] - dLj_Middles[:,y]
            # Save the norm of each vector
            r_ij_norm[x,y] = np.linalg.norm(r_ij[x,y,:])
            r_ji_norm[y,x] = np.linalg.norm(r_ji[y,x,:])
            # Make the vectors connecting i and j into unit vectors
            r_ij_unit[x,y,:] = r_ij[x,y,:] / r_ij_norm[x,y]
            r_ji_unit[y,x,:] = r_ji[y,x,:] / r_ji_norm[y,x]


    # Compute Forces and Torques
    IidLi = np.zeros((3,Ni))
    IjdLj = np.zeros((3,Nj))
    IidLi= Ii*dLi_Vectors
    IjdLj= Ij*dLj_Vectors
    Fij = 0
    Fji = 0
    Tij = 0
    Tji = 0

    # Compute cross products for all j and i. Rather than computing all 4 values (dF and dT and incrementing F and T) all within the nested loop, we compute the cross products that requiring the nested j and store in matrix TMPj. Now can cross the IidLi with this matrix, rather than crossing every vector separately. This vectorisation led to 2x speed-up.
    TMPj= np.zeros((3,Nj))
    for i in range (Ni):
        for j in range(Nj):
            TMPj[:,j] = np.cross(IjdLj[:,j] , r_ji_unit[j,i,:]) / (r_ji_norm[j,i]**2)
        dF_ij = np.cross(IidLi[:,i] , TMPj,axis=0)
        dT_ij = np.cross(R_i[:,i] , dF_ij,axis=0)
        Fij = Fij + np.sum(dF_ij,axis=1)
        Tij = Tij + np.sum(dT_ij,axis=1)

    TMPi= np.zeros((3,Ni))
    for j in range (Nj):
        for i in range(Ni):
            TMPi[:,i] = np.cross(IidLi[:,i] , r_ij_unit[i,j,:]) / (r_ij_norm[i,j]**2)
        dF_ji = np.cross(IjdLj[:,j] , TMPi,axis=0)
        dT_ji = np.cross(R_j[:,j] , dF_ji,axis=0)
        Fji = Fji + np.sum(dF_ji,axis=1)
        Tji = Tji + np.sum(dT_ji,axis=1)

    F_12 = MuO/(4*pi)* Fij
    F_21 = MuO/(4*pi)* Fji
    T_12 = MuO/(4*pi)* Tij
    T_21 = MuO/(4*pi)* Tji

    # Print Forces, Torques and time taken
    t1 = time.clock()
    time_comp= t1 - t0
    z_sep= (d[2]-(turns_i*h_i*2*pi))*1000
    y_sep= (d[1]-(radius_i+radius_j))*1000

    #Plot
    PlotWires(dLi_Nodes,dLj_Nodes,dLi_Middles,dLj_Middles)

    ############# Printing ##############
    '''if (ChooseParallelLines==True) and (ChooseCoils==False):
        print ('Ni\t',Ni,'\nLength\t',X_length,'\nComp time\t',time_comp,'\nF12/L \t',F_12/X_length,'\nF21/L\t',F_21/X_length,'\nT12 \t',T_12,'\nT21\t',T_21)
        # By the definition of the Ampere: For infinite parallel wires, placed 1m apart, for a 1A current in each in same direction, the force on each should equal 2*10e-7 N in attraction.
        print('\nBy the definition of the Ampere: For infinite parallel wires, placed 1m apart, for a 1A current in each in same direction, the force/length on each should equal 2*10e-7 N in attraction')
    elif (ChooseParallelLines==False) and (ChooseCoils==True):
        # if coils are axially aligned (separated in Z)
        if (d[0]==0) and (d[1]==0) and (d[2]>0):
            print ('Ni\t',Ni,'\nCoil sep. XYZ (mm)\t',d*1000,'\nClosest sep. in Z (mm)\t',z_sep,'\nComp time (s)\t',time_comp,'\nF12\t',F_12,'\nF21\t',F_21,'\nT12 \t',T_12,'\nT21\t',T_21)
            print('\nLin Acc of in Z of 1 cube between 2x250g cubes using 1 pair of EMs:\n',(F_12[2]/0.25)*1000*1*1,'mm/s/s')
            # Call FarField Model for comparison
            FarFieldModel(d)
        # elif coils are parallel (separated in Y), change output
        elif (d[0]==0) and (d[1]>0) and (d[2]==0):
            print ('Ni\t',Ni,'\nCoil sep. XYZ (mm)\t',d*1000,'\nClosest sep. in Y (mm)\t',y_sep,'\nComp time (s)\t',time_comp,'\nF12\t',F_12,'\nF21\t',F_21,'\nT12 \t',T_12,'\nT21\t',T_21)
            print('\nLin Acc in Y of 1 cube between 2x250g cubes using 1 pair of EMs:\n',(F_12[1]/0.25)*1000*1*1,'mm/s/s')
        else:
            print('\ncoil separation is not non-zero in just Y or just Z')
    else:
        print('\nWARNING: You can only select one geometry to compute\n')
    '''




    ####################### Debugging ComputeForcesAndTorques() ###########
    Debug= False
    if Debug==True:

    #For r: Check the vector magnitude in the x, y and z directions of r
        print('r_ij and r_ji \n')
        for x in range(3):
            print(r_ij[:,:,x])
            print(r_ji[:,:,x], '\n\n')
    #For r: check the vector stored along z axis of r
        print('norm of r_ij and r_ji \n',np.linalg.norm(r_ij, axis=2),'\n',np.linalg.norm(r_ji, axis=2),'\n\n')

    #For r_unit: Check the vector magnitude in the x, y and z directions of r
        print('r_ij_unit and r_ji_unit \n')
        for x in range(3):
            print(r_ij_unit[:,:,x])
            print(r_ji_unit[:,:,x], '\n\n')
    #For r_unit: After normalizing to unit vector, check the vector stored along z axis of r has mag 1
        print('norm of r_ij_unit and r_ji_unit \n',np.linalg.norm(r_ij_unit, axis=2),'\n',np.linalg.norm(r_ji_unit, axis=2),'\n\n')

    # Check Middles and Vectors
        print('dLi Middles \n',dLi_Middles,' \ndLj Middles \n',dLj_Middles,'\ndLi Vectors', dLi_Vectors,'\ndLj Vectors \n',dLj_Vectors)

    return F_12,F_21,T_12,T_21,time_comp,y_sep,z_sep


# Run main
main()
