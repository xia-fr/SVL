import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from Core.PlaneWave import *

def GetTimeSeries(vals, dt, option):
    '''
    This function computes the missing time series for the displacement, 
    velocity and acceleration depending on the option provied by the user\n
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Danilo S. Kusanovic 2021

    Parameter
    ----------
    vals   : array
        The provided time series
    dt     : float
        The time step for the given time series 
    option : str
        User's time series data, option=DISP, VEL, or ACCEL

    Returns
    -------
    Disp, Vels, Accel the time series
    '''
    #Compute other fields according to option
    if  option.upper() == 'DISP':
        Disp  = vals
        Vels  = GetDerivative(vals, dt)
        Accel = GetDerivative(Vels, dt)
    elif option.upper() == 'VEL':
        Disp  = GetIntegration(vals, dt)
        Vels  = vals
        Accel = GetDerivative (vals, dt)
    elif option.upper() == 'ACCEL':
        Vels  = GetIntegration(vals, dt)
        Disp  = GetIntegration(Vels, dt)
        Accel = vals
    
    return Disp, Vels, Accel

def Compute2DFreeFieldBoundaries(nodes, boundSide, signal, option, dt, xmin, layers, beta_layer, rho_layer, nu_layer, th, df=0.2, cof=30.0):
    '''
    This function calculates the free-field forces required to be applied at the boundaries.\n
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Feiruo (Flora) Xia, Eugene Loh, and Yaozhong Shi
    
    Parameters
    ----------
    nodes  : array
        Contains coordinates of all nodes in the model and their nid.
        i.e. nodes = [[x0, y0, nid0], [x1, y1, nid1], ...]
    boundSide : string
        'left' or 'right' for which boundary side. Also names the files in which to save node reaction data. 
        e.g. boundSide = 'left' will produce left_y_5.in for the vertical reaction at nid 5.
    signal  : array
        Vector with the time series values used as an input. The type of time series is determined by 'option'.
    option  : string
        Defines the type of input that 'signal' is, either 'DISP' for displacement, 'VEL' for velocity, or
        'ACCEL' for acceleration.
    dt  : float
        The time-step size of the time-series.
    xmin : array
        Coordinates (x and y) of the lowest node of the boundary.
    layers : array
        Y-coordinates of nodes at layer boundaries, including the top of the top layer and the half space.
        (in descending order as well)
    beta_layer, rho_layer, nu_layer  : array
        Shear wave velocity, mass density, and Poisson's ratio of soil layers, from top layer to half space (e.g. descending order).
    th : thickness

    Returns
    -------
    cs_all  : array
        Calculated shear dashpot coefficient for each node on the boundary in the order given by input nodes.
    cp_all  : array
        Calculated perpendicular dashpot coefficient for each node on the boundary in the order given by input nodes.

    Saves
    -------
    Vertical and horizontal reactions for nodes on the boundary, numbered by node id.
    '''

    # Sign adjustment depending on which side the free-field boundary is on
    if boundSide.upper() == 'LEFT':
        sign = -1
    else:
        sign = 1

    # Finds user's absolute path for the file that is running this function
    filePath = os.path.abspath(os.path.dirname(sys.argv[0]))
    os.makedirs(filePath + '/freefield_reactions', exist_ok=True)

    # Unpack predefined SV-Wave 
    angle = 0.0001
    y0 = xmin[1]
    x0 = xmin[0]

    # Sorts relevant arrays to proper order
    nodes = nodes[nodes[:,1].argsort()] # ascending
    layers = layers[layers.argsort()[::-1]] # descending

    # Creates the fun dictionary required for DataPreprocessing
    fun = { 'option': 'SV', 'df': df, 'CutOffFrequency': cof}

    # Computes the input displacement, velocities and accelerations.
    Disp, Vels, Accel = GetTimeSeries(signal, dt, option)

    # Transform time-series in frequency domain and computes variables required to compute reponses layer interface
    nt = len(signal)
    ufull, vfull, afull, layers, beta_layer, rho_layer, nu_layer, wVec, p, s, h, mu, aSP, phaseVelIn, sinTheta, N, Nt = DataPreprocessing(Disp, Vels, Accel, layers, beta_layer, rho_layer, nu_layer, angle, y0, nt, dt, fun)

    # Compute Interface responses
    uInterface = SoilInterfaceResponse(ufull, wVec, p, s, h, mu, aSP, phaseVelIn, sinTheta, N)
    vInterface = SoilInterfaceResponse(vfull, wVec, p, s, h, mu, aSP, phaseVelIn, sinTheta, N)

    # Extracts x and y values from nodes
    x = nodes[:,0]
    y = nodes[:,1]
    id = nodes[:,2]

    # Get moduli for each element from layer data (ascending order)
    beta = np.zeros(len(y)-1)
    rho = np.zeros(len(y)-1)
    nu = np.zeros(len(y)-1)
    for gx in range(len(y)-1):
        beta += beta_layer[gx] * (y[:-1]<layers[gx])*(y[:-1]>=layers[gx+1])
        rho += rho_layer[gx] * (y[:-1]<layers[gx])*(y[:-1]>=layers[gx+1])
        nu += nu_layer[gx] * (y[:-1]<layers[gx])*(y[:-1]>=layers[gx+1])
    G = rho*beta**2
    beta_p = beta * np.sqrt(2*(1 - nu)/(1-2*nu))

    # Initialize arrays
    cs_all = np.zeros(len(x))
    cp_all = np.zeros(len(x))
    U = np.zeros((len(x), Nt))
    V = np.zeros((len(x), Nt))
    vertReact = np.zeros((len(x), Nt))
    horizReact = np.zeros((len(x), Nt))
    
    # Calculate horizontal displacement and velocity (ascending based on x,y)
    for ix in range(len(x)):
        Utemp = PSVbackground2Dfield(uInterface, layers, wVec, p, s, h, mu, aSP,\
                                    phaseVelIn, sinTheta, N, Nt, x0, x[ix], y[ix])
        Vtemp = PSVbackground2Dfield(vInterface, layers, wVec, p, s, h, mu, aSP,\
                                    phaseVelIn, sinTheta, N, Nt, x0, x[ix], y[ix])
        U[ix] = Utemp[:,0]
        V[ix] = Vtemp[:,0]
    
    # Calculate force contribution of each element
    dh = y[1:]-y[:-1]
    trib_area = dh*th/2
    dU = U[1:]-U[:-1]
    F = np.transpose(np.transpose(dU)*G/dh*trib_area)

    # Calculate vertical reactions and save
    vertReact[0] = sign*F[0]
    vertReact[1:-1] = sign*(F[:-1] + F[1:])
    vertReact[-1] = sign*F[-1]
    for ix, react in enumerate(vertReact):
        np.savetxt(filePath+'/freefield_reactions/'+boundSide+'_y_'+f'{id[ix]:.0f}'+'.in',\
                np.append([0.0], react),\
                fmt='%.8e',\
                header=str(Nt+1), comments='',\
                newline='\n')
        if ix == len(vertReact)-1:
            print(boundSide.upper()+' side vertical reactions: '+str(ix+1)+' out of '+str(len(nodes)))
        else:
            print(boundSide.upper()+' side vertical reactions: '+str(ix+1)+' out of '+str(len(nodes)), end='\r', flush=True)
    
    # Calculate dashpot coefficients
    cs = rho*beta*trib_area
    cs_all[1:-1] = cs[:-1] + cs[1:]
    cs_all[-1] = cs[-1]
    cs_all[0] = cs[0]

    cp = rho*beta_p*trib_area
    cp_all[1:-1] = cp[:-1] + cp[1:]
    cp_all[-1] = cp[-1]
    cp_all[0] = cp[0]

    # Save horizontal velocity
    horizReact = V
    for ix, react in enumerate(horizReact):
        np.savetxt(filePath+'/freefield_reactions/'+boundSide+'_horizVel_'+f'{id[ix]:.0f}'+'.in',\
                np.append([0.0], react),\
                fmt='%.8e',\
                header=str(Nt+1), comments='',\
                newline='\n')
        if ix == len(horizReact)-1:
            print(boundSide.upper()+' side horizontal velocities: '+str(ix+1)+' out of '+str(len(nodes)))
            print('')
        else:
            print(boundSide.upper()+' side horizontal velocities: '+str(ix+1)+' out of '+str(len(nodes)), end='\r', flush=True)

    # DEBUGGING:
    for ix, disp in enumerate(U):
        np.savetxt(filePath+'/freefield_reactions/'+boundSide+'_U_'+f'{id[ix]:.0f}'+'.in',\
                np.append([0.0], disp),\
                fmt='%.8e',\
                header=str(Nt+1), comments='',\
                newline='\n')

    return cs_all, cp_all

def Compute3DFreeFieldBoundaries(x, y, signal, dt, xmin, layers, beta_layer, rho_layer, nu_layer, option, df=0.2, cof=30.0):
    '''
    '''
    pass

def Find2DBoundaries(nodes):
    '''
    This function finds the nodes at the left, right, and bottom boundaries of the given model,
    given the coordinates of all the nodes, and their corresponding node ids.

    Note that this assumes the left, right, and bottom boundaries of the model are perfectly square
    with the coordinate frame.

    @author Feiruo (Flora) Xia, Eugene Loh, and Yaozhong Shi
    Parameters
    ----------
    nodes  : array
        Contains coordinates of all nodes in the model and their nid.
        i.e. nodes = [[x0, x1, ...], [y0, y1, ...], [nid0, nid1, ...]]

    Returns
    -------
    leftNodes   : array
        Contains xy pairs of all nodes on the left boundary of given model sorted from low to high in y.
        The third value attached to each pair is the node ID.
    rightNodes   : array
        Contains xy pairs of all nodes on the right boundary of model sorted from low to high in y.
        The third value attached to each pair is the node ID.
    bottomNodes   : array
        Contains xy pairs of all nodes on the bottom boundary of model sorted from left to right in x.
        The third value attached to each pair is the node ID.
    '''

    # Extract x and y node coordinates from array, and nid
    x = nodes[0]
    y = nodes[1]
    nids = nodes[2]

    # Finds the coordinates of the left, right, and bottom boundaries
    left = min(x)
    right = max(x)
    bottom = min(y)
    tol = 1.0E-03

    leftNodes, rightNodes, bottomNodes = list(), list(), list()
    for ind, nid in enumerate(nids):
        if np.abs(bottom - y[ind]) < tol:
            bottomNodes.append([x[ind], y[ind], nid])

        if np.abs(right - x[ind]) < tol:
            rightNodes.append([x[ind], y[ind], nid])

        if np.abs(left - x[ind]) < tol:
            leftNodes.append([x[ind], y[ind], nid])

    # Sorts node lists by relevant coordinate
    rightNodes.sort(key=lambda x:x[1])
    leftNodes.sort(key=lambda x:x[1])
    bottomNodes.sort(key=lambda x:x[0])

    return np.array(leftNodes), np.array(rightNodes), np.array(bottomNodes)