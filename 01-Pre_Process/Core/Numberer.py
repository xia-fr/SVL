#!/usr/bin/python3
# -*- coding: Utf-8 -*-

import random
import numpy as np
import scipy.sparse as sps
import matplotlib.pylab as plt
from scipy.sparse import find
from scipy.sparse import linalg as sla
from Core.Definitions import Entities, Options

def PetscAllocation():
    """
    This function computes (bruta fuerza) the number of non-zero for each 
    diagonal and off-diagonal blocks required for allocation in PETSc. In
    simple words, it estimates the memory allocation for PETSc.\n
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Danilo S. Kusanovic 2020

    Parameters
    ----------
    None

    Output
    -------
    None
    """
    N = Options['nparts']
    M = Options['nfree']

    nr = int(M/N)
    n_nz = np.zeros(N, dtype=int)
    o_nz = np.zeros(N, dtype=int)

    #Assembles the matrix.
    A = FormSparseMatrix()

    #Re-define matrix.
    (I,J,V) = find(A)
    V = 0.0*V + 1.0
    A = sps.coo_matrix((V,(I, J)), shape=(M,M))
    A = A.tocsr()

    #Loop over the partitions.
    for k in range(0,N):
        #Matrix partition.
        inf = k*nr
        sup = (k + 1)*nr

        #Sub-Matrix stripe. 
        S = A[inf:sup,:]

        #Sub-Matrix block-diagonal.
        D = S[:,inf:sup]

        #Check the partition is not empty
        if D.size != 0:
            n_nz[k] = np.max(D.sum(axis=1))
            o_nz[k] = np.max(S.sum(axis=1) - D.sum(axis=1))

    #Sets the closest multipple of five.
    n_nz = np.floor( np.divide(n_nz,5) + 1) * 5
    o_nz = np.floor( np.divide(o_nz,5) + 1) * 5

    Options['d_nz'] = n_nz
    Options['o_nz'] = o_nz

def plotSparsePattern(A):
    """
    This function display the sparsity pattern of a given matrix in
    sparse format (COO, CRS).\n
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Danilo S. Kusanovic 2020

    Parameters
    ----------
    A : The sparse matrix to be displayed

    Output
    -------
    None
    """
    filepath = Options['path'] + '/' + 'MatrixPattern.png'

    plt.spy(A, markersize=2)
    plt.savefig(filepath)
    plt.grid(True)

    plt.close()

def FormSparseMatrix(format='coo'):
    """
    This function compute/emulates the Sparse Matrix Pattern to be employed 
    in the user's defined ordering scheme.\n
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Danilo S. Kusanovic 2020

    Parameters
    ----------
    format : str
        The sparse matrix format: coo, csr, csc

    Output
    -------
    None
    """
    #Emulates the Element Assembly Pattern 
    N = Options['ntotal']
    M = Options['nconsistent']

    I = np.zeros(M, dtype=int)
    J = np.zeros(M, dtype=int)
    V = np.zeros(M, dtype=float)

    m = 0
    for eTag in Entities['Elements']:
        #Element's node connectivity array
        connection = Entities['Elements'][eTag]['conn']

        #Element's degree-of-freedom list
        total = []
        for nTag in connection:
            total = np.concatenate((total, Entities['Nodes'][nTag]['totaldof']))

        for idof in total:
            for jdof in total:
                I[m] = idof
                J[m] = jdof
                V[m] = random.random()
                m += 1

    A = sps.coo_matrix((V,(I, J)), shape=(N,N))

    #Emulates the Transformation matrix 
    M = Options['nfree']
    N = Options['ntotal']
    S = Options['nfree'] + Options['nconstraint']

    I = np.zeros(S, dtype=int)
    J = np.zeros(S, dtype=int)
    V = np.zeros(S, dtype=float)

    m = 0
    for nTag in Entities['Nodes']:
        ndof  = Entities['Nodes'][nTag]['ndof' ]
        free  = Entities['Nodes'][nTag]['freedof' ]
        total = Entities['Nodes'][nTag]['totaldof']

        for k in range(ndof):
            #The Free Degree-Of-Freedom is unconstrained
            if free[k] > -1:
                I[m] = total[k]
                J[m] = free[k]
                V[m] = 1.00
                m += 1

            #The Free Degree-Of-Freedom is constrained
            if free[k] < -1:
                SlaveNode = Entities['Constraints'][free[k]]['stag']
                SlaveDOF  = Entities['Constraints'][free[k]]['sdof' ]
                Slave     = Entities['Nodes'][SlaveNode]['totaldof'][SlaveDOF]

                MasterNode = Entities['Constraints'][free[k]]['mtag']
                MasterDOF  = Entities['Constraints'][free[k]]['mdof']

                for i in range(len(MasterNode)):
                    Master = Entities['Nodes'][MasterNode[i]]['freedof'][MasterDOF[i]]
                    I[m] = Slave
                    J[m] = Master
                    V[m] = 1.00 
                    m += 1

    T = sps.coo_matrix((V,(I, J)), shape=(N,M))

    #The Final Element Assembly pattern 
    A = T.transpose()*A*T

    if format == 'csr':
        A = A.tocsr()
    elif format == 'csc':
        A = A.tocsc()

    return A

def PlainScheme():
    """
    This function assign a Plain Scheme to the degree of freedom of each Node, 
    i.e., the node numbering is assigned consecutively from the lowest to the 
    highest Node identifier.\n
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Danilo S. Kusanovic 2020

    Parameters
    ----------
    None

    Output
    -------
    None
    """
    #Counter Variables
    count0 = 0
    count1 = 0
    nConstraintDofs = 0

    #Assign consecutively degree of freedom numbering
    for nTag in sorted(list(Entities['Nodes'].keys())):
        ndof = Entities['Nodes'][nTag]['ndof' ]
        free = Entities['Nodes'][nTag]['freedof' ]
        total = Entities['Nodes'][nTag]['totaldof']

        #Total degree-of-freedom numbering.
        for k in range(ndof):
            total[k] = count0
            count0 += 1

        #Free degree-of-freedom numbering.
        for k in range(ndof):
            if free[k] > -1:
                free[k] = count1
                count1 += 1
            elif free[k] < -1:
                MasterNodes = Entities['Constraints'][free[k]]['mtag'] 
                nConstraintDofs += len(MasterNodes)

        #Assign ne Free/Total degree-of-freedom numbering.
        Entities['Nodes'][nTag]['freedof' ] = free
        Entities['Nodes'][nTag]['totaldof'] = total

    #Write number of Free/Total/Constrained degree-of-freedom.
    Options['nfree'] = count1
    Options['ntotal'] = count0
    Options['nconstraint'] = nConstraintDofs

def MinimumDegreeScheme():
    """
    This function performs the Minimum-Degree ordering Scheme to the free
    degree of freedom. First compute the Sparse Matrix Pattern, and then
    computes the permutation vector. Such vector is finally employed to
    re-label the free-degree-of-freedom.\n
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Danilo S. Kusanovic 2020

    Parameters
    ----------
    None

    Output
    -------
    None
    """
    #Gets the Sparse Matrix to Perform Permutation
    A = FormSparseMatrix('csc')

    #Computes the Permutation Vector for the Matrix
    lu = sla.splu(A, permc_spec='MMD_ATA')
    perm = lu.perm_c

    #The New Free-Degree-Of-Freedom Numbering
    N = len(perm)
    I = np.zeros(N, dtype=int)
    I[perm] = np.arange(0, N, 1)

    #Transform the degree of freedom numbering form Plain to CutHill-McKee
    for nTag in Entities['Nodes']:
        ndof = Entities['Nodes'][nTag]['ndof']
        free = Entities['Nodes'][nTag]['freedof']
        aux  = np.zeros(ndof, dtype=int)  

        #New-Free degree-of-freedom numbering.
        for k in range(ndof):
            if free[k] > -1:
                aux[k] = I[free[k]]
            else:
                aux[k] = free[k]

        #Assign the Free degree-of-freedom numbering.
        Entities['Nodes'][nTag]['freedof'] = aux
    print('\x1B[33m ALERT \x1B[0m: The Minimum Degree scheme has not been validated.')

def CutHillMcKeeScheme():
    """
    This function performs the CutHill-McKee ordering Scheme to the free
    degree of freedom. First compute the Sparse Matrix Pattern, and then
    computes the permutation vector. Such vector is finally employed to
    re-label the free-degree-of-freedom.\n
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Danilo S. Kusanovic 2020

    Parameters
    ----------
    None

    Output
    -------
    None
    """
    #Gets the Sparse Matrix to Perform Permutation
    A = FormSparseMatrix('csr')

    #Computes the Permutation Vector for the Matrix
    perm = sps.csgraph.reverse_cuthill_mckee(A, True)

    #The New Free-Degree-Of-Freedom Numbering
    N = len(perm)
    I = np.zeros(N, dtype=int)
    I[perm] = np.arange(0, N, 1)

    #Transform the degree of freedom numbering form Plain to CutHill-McKee
    for nTag in Entities['Nodes']:
        ndof = Entities['Nodes'][nTag]['ndof']
        free = Entities['Nodes'][nTag]['freedof']
        aux  = np.zeros(ndof, dtype=int)  

        #New-Free degree-of-freedom numbering.
        for k in range(ndof):
            if free[k] > -1:
                aux[k] = I[free[k]]
            else:
                aux[k] = free[k]

        #Assign the Free degree-of-freedom numbering.
        Entities['Nodes'][nTag]['freedof'] = aux

#Finds nodes that do not belong elements
def FindDefectiveNodes():
    """
    This function loops over the Element and identify Nodes that do not
    belong to them. Then, it fix such Nodes that are defective. Constraints 
    of type Diaphragm or General are not considered defective.\n
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Danilo S. Kusanovic 2020

    Parameters
    ----------
    None

    Output
    -------
    None
    """
    #Memory Storage Variables
    nLumpedStorage = 0
    nConsistentStorage = 0

    #Identifies Nodes which don't belong to Element.
    Condition = dict.fromkeys(Entities['Nodes'].keys(), True)
    for eTag in Entities['Elements']:
        nDOFelem   = 0
        connection = Entities['Elements'][eTag]['conn']
        for conn in connection:
            Condition[conn] = False
            nDOFelem += Entities['Nodes'][conn]['ndof']

        nLumpedStorage     += nDOFelem
        nConsistentStorage += nDOFelem*nDOFelem

    #Diaphragm, Equal, and General Constraints Nodes are considered non-defective.
    for cTag in  Entities['Constraints']:
        mNodes = Entities['Constraints'][cTag]['mtag']
        for mTag in mNodes:
            Condition[mTag] = False

    #Fix all Detected Defective Nodes
    for nTag in Condition:
        if Condition[nTag]:
            ndof = Entities['Nodes'][nTag]['ndof']
            if np.sum(Entities['Nodes'][nTag]['freedof']) != -ndof:
                Entities['Nodes'][nTag]['freedof'] = np.full(ndof, -1, dtype=int)
                print('\x1B[33m ALERT \x1B[0m: The Node[' + str(nTag) + '] does not belong to an element. Node will be fixed.')

    #Saves in Options the Matrix Memory Storage 
    Options['nlumped'] = nLumpedStorage
    Options['nconsistent'] = nConsistentStorage

def setDegreeOfFreedom(plot=False):
    """
    This function assigns the degree of freedom numbering for each Node 
    according to the User's numbering pattern.\n
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Danilo S. Kusanovic 2020

    Returns
    -------
    None
    """
    #Detects Point that does not belong to Element
    FindDefectiveNodes()

    #Assign Plain Numbering Scheme
    PlainScheme()

    #Impose User's Define Numbering Scheme
    if Options['numbering'].upper() == 'PLAIN':
        pass
    elif Options['numbering'].upper() == 'CUTHILL-MCKEE':
        CutHillMcKeeScheme()
    elif Options['numbering'].upper() == 'MINIMUM DEGREE':
        MinimumDegreeScheme()

    #Gets the Sparse Matrix to Perform Permutation
    if plot:
        A = FormSparseMatrix()
        plotSparsePattern(A)

    #Compute memory storage for Petsc matrix.
    if Options['allocation'] == 'YES':
        PetscAllocation()
