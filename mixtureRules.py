# -*- coding: utf-8 -*-
"""

Reduced-pressure-based mixture rules for evaluting multi-component pressure dependence.

Reference:
    [1] Lei Lei, Michael P. Burke. "Bath gas mixture effects on multichannel reactions: In-sights and representations 
        for systems beyond single-channel reactions". J. Phy. Chem.A, 123(3):631-649, 2019.
    [2] Lei Lei, Michael P. Burke. "Evaluating the performance of bath gas mixture rules forgeneral implementation in 
    	chemically reacting flow codes: Tests for multi-well, multi-channel reactions". 12th U.S. National Combustion Meeting, 
    	College Station, Texas, USA, May 2021.

@author: Lei Lei
"""

import numpy as np
import types, sys
from sympy.solvers import solve
from sympy import Symbol
from sympy.matrices import *

######################################################

def LMR_R(T, P, X, eig_0, k_P):
    """
        Linear mixture rule in reduce pressure for a mixture consisting of M components.
        Input:
            T: float, local temperature, K;
            P: float, local pressure, atm;
            X: list of shape (M,), local mixture composition;
            eig_0: list of shape (M,), absolute value of least negative chemically significant
                   eigenvalue in the low-pressure limit for each component;
            k_P: list of 2d-arraies or functions, used to evaluate channel-specific rate coefficients
                 at given T,P. If list of 2d-arraies, each 2d-array should have pressure and the
                 corresponding rate coefficient at considered temperature; if list of functions,
                 each function should take T,P and calculate rate coefficient.
        Ouput:
            k_LMR, float, rate coefficients evaluated using LMR,R.
    """

    eig_0_mix = np.sum(np.array(X) * np.array(eig_0))
    P_i = P * eig_0_mix / np.array(eig_0)
    x_tilde = np.array(eig_0) * np.array(X) / eig_0_mix

    temp = []
    # calculate rate coefficients at reuduce pressure
    for i,p in enumerate(P_i):
        if isinstance(k_P[i], types.FunctionType):
            try:
                temp.append(k_P[i](T,p))
            except:
                sys.exit("Function k_P(T,P) takes two variables.")
        # if 2d-arraies are used, interpolate using PLOG formula
        elif type(k_P[i]) == np.ndarray:
            # locate the pressure of interest
            P_loc = np.sum(k_P[i][0,:] < p)
            # interpolate using PLOG form
            k_temp = 10 ** (np.log10(k_P[i][1,P_loc]) + (np.log10(k_P[i][1,P_loc+1]) - np.log10(k_P[i][1,P_loc])) /
                           (np.log10(k_P[i][0,P_loc+1]) - np.log10(k_P[i][0,P_loc])) * (np.log10(P) + np.log10(eig_0_mix) - np.log10(k_P[i][0,P_loc]) - np.log10(eig_0[i])))
            temp.append(k_temp)
    # wrighted sum to get LMR,R
    k_LMR = np.sum(np.array(temp) * x_tilde)

    return k_LMR

def activity_coefficient(T, X, Z, delta_e_d, Fe):
    """
        Solve for the activity coefficients for a single-well system in a mixture consisting of M components.
        Input:
            T: float, local temperature, K;
            P: float, local pressure, atm;
            X: list of shape (M,), local mixture composition;
            Z: list of shape (M,), collision frequences of each component, s-1;
            delta_e_d: list of shape (M,), averaged energy transferred per downward collision, cm-1;
            Fe: float, energy dependence of the density of states of the complex near the lowest decomposition threshold.
        Output:
            f: list of shape (M,), activity coefficients at the local conditions.
    """

    k_B = 0.695034800 # Boltzmann constant in cm-1 K-1
    delta_e_u = np.array(delta_e_d) * Fe * k_B * T / (np.array(delta_e_d) + Fe * k_B * T)
    x_hat = np.array(X) * np.array(Z) / np.sum(np.array(X) * np.array(Z))

    D = Symbol('D')
    # Create equation for D's
    f_D = lambda D: -np.sum([x_hat[i] * D ** 2 / ((delta_e_d[i] + D) * (delta_e_u[i] - D)) for i in range(len(X))])
    # solve for the equation
    D_step = np.array([i.as_real_imag()[0] for i in solve(f_D(D) - 1., D)])
    # drop the negative results
    D_step = D_step[D_step >= 0]

    if len(D_step) == len(X):
        coff_matrix = Matrix([[D_step[i] / (delta_e_u[j] - D_step[i]) for i in range(len(X))] for j in range(len(X))])
        C_step = coff_matrix.inv()
        C_step = C_step.dot(Matrix(-1. * np.ones(len(X))))
    # if there are zeros in the X array
    else:
        temp = np.zeros(X.shape)
        temp_C = np.zeros(X.shape)
        temp[X>0] = D_step[:]
        temp_D = temp[X>0]
        D_step = temp
        temp_belta = delta_e_u[X>0]
        coff_matrix = Matrix([[temp_D[i] / (temp_belta[j] - temp_D[i]) for i in range(len(temp_D))] for j in range(len(temp_D))])
#        print(coff_matrix)
        C_step = coff_matrix.inv()
        temp_C[X>0] = C_step.dot(Matrix(-1. * np.ones(len(X)-1)))
        C_step = temp_C

    f_list = (delta_e_d + Fe * k_B * T) / (delta_e_d + delta_e_u) * (1. - np.array([np.sum(C_step * D_step / (delta_e_d[i] + D_step)) for i in range(len(X))]))
    return [float(x) for x in f_list]

def NMR_R(T, P, X, eig_0, f, k_P):
    """
        Nonlinear mixture rule in reduce pressure for a mixture consisting of M components.
        Input:
            T: float, local temperature, K;
            P: float, local pressure, atm;
            X: list of shape (M,), local mixture composition;
            eig_0: list of shape (M,), absolute value of least negative chemically significant
                   eigenvalue in the low-pressure limit for each component;
            f: list of shape (M,), activity coefficients for each component;
            k_P: list of 2d-arraies or functions, used to evaluate channel-specific rate coefficients
                 at given T,P. If list of 2d-arraies, each 2d-array should have pressure and the
                 corresponding rate coefficient at considered temperature; if list of functions,
                 each function should take T,P and calculate rate coefficient.
        Ouput:
            k_NMR, float, rate coefficients evaluated using NMR,R.
    """

    eig_0_mix = np.sum(np.array(X) * np.array(eig_0) * np.array(f))
    P_i = P * eig_0_mix / (np.array(eig_0))
    x_tilde = np.array(eig_0) * np.array(X) * np.array(f) / eig_0_mix

    temp = []
    # calculate rate coefficients at reuduce pressure
    for i,p in enumerate(P_i):
        if isinstance(k_P[i], types.FunctionType):
            try:
                temp.append(k_P[i](T,p))
            except:
                sys.exit("Function k_P(T,P) takes two variables.")
        # if 2d-arraies are used, interpolate using PLOG formula
        elif type(k_P[i]) == np.ndarray:
            # locate the pressure of interest
            P_loc = np.sum(k_P[i][0,:] < p)
            # interpolate using PLOG form
            k_temp = 10 ** (np.log10(k_P[i][1,P_loc]) + (np.log10(k_P[i][1,P_loc+1]) - np.log10(k_P[i][1,P_loc])) /
                           (np.log10(k_P[i][0,P_loc+1]) - np.log10(k_P[i][0,P_loc])) * (np.log10(P) + np.log10(eig_0_mix) - np.log10(k_P[i][0,P_loc]) - np.log10(eig_0[i])))
            temp.append(k_temp)
    # wrighted sum to get NMR,R
    k_NMR = np.sum(np.array(temp) * x_tilde)

    return k_NMR

###############################################################################
if __name__ == '__main__':
    # LMR,R implementation
    T = 700. # K
    P = 1.   # atm
    X = [0.9, 0.1] # mole fraction for a mixture consisting of 2 components
    eig_0 = [2.32467020E-07, 3.70785356E-06] # absolute values of least negative chemically significant eigenvalues in the low-pressure limit for each components

    # specify channel-specific rate coefficients for each components
    # note that k_P_i can also be functions k_P_i(T,P) that takes in (temperature, pressure) pair as aguement
    k_P_1 = np.array([[1.00E-14, 2.00E-14, 3.00E-14, 4.00E-14, 5.00E-14, 6.00E-14, 7.00E-14, 8.00E-14, 9.00E-14, 1.00E-13, 2.00E-13, 3.00E-13, 4.00E-13, 5.00E-13, 6.00E-13, 7.00E-13, 8.00E-13, 9.00E-13, 1.00E-12, 2.00E-12, 3.00E-12, 4.00E-12, 5.00E-12, 6.00E-12, 7.00E-12, 8.00E-12, 9.00E-12, 1.00E-11, 2.00E-11, 3.00E-11, 4.00E-11, 5.00E-11, 6.00E-11, 7.00E-11, 8.00E-11, 9.00E-11, 1.00E-10, 2.00E-10, 3.00E-10, 4.00E-10, 5.00E-10, 6.00E-10, 7.00E-10, 8.00E-10, 9.00E-10, 1.00E-09, 2.00E-09, 3.00E-09, 4.00E-09, 5.00E-09, 6.00E-09, 7.00E-09, 8.00E-09, 9.00E-09, 1.00E-08, 2.00E-08, 3.00E-08, 4.00E-08, 5.00E-08, 6.00E-08, 7.00E-08, 8.00E-08, 9.00E-08, 1.00E-07, 2.00E-07, 3.00E-07, 4.00E-07, 5.00E-07, 6.00E-07, 7.00E-07, 8.00E-07, 9.00E-07, 1.00E-06, 2.00E-06, 3.00E-06, 4.00E-06, 5.00E-06, 6.00E-06, 7.00E-06, 8.00E-06, 9.00E-06, 1.00E-05, 2.00E-05, 3.00E-05, 4.00E-05, 5.00E-05, 6.00E-05, 7.00E-05, 8.00E-05, 9.00E-05, 1.00E-04, 2.00E-04, 3.00E-04, 4.00E-04, 5.00E-04, 6.00E-04, 7.00E-04, 8.00E-04, 9.00E-04, 1.00E-03, 2.00E-03, 3.00E-03, 4.00E-03, 5.00E-03, 6.00E-03, 7.00E-03, 8.00E-03, 9.00E-03, 1.00E-02, 2.00E-02, 3.00E-02, 4.00E-02, 5.00E-02, 6.00E-02, 7.00E-02, 8.00E-02, 9.00E-02, 1.00E-01, 2.00E-01, 3.00E-01, 4.00E-01, 5.00E-01, 6.00E-01, 7.00E-01, 8.00E-01, 9.00E-01, 1.00E+00, 2.00E+00, 3.00E+00, 4.00E+00, 5.00E+00, 6.00E+00, 7.00E+00, 8.00E+00, 9.00E+00, 1.00E+01, 2.00E+01, 3.00E+01, 4.00E+01, 5.00E+01, 6.00E+01, 7.00E+01, 8.00E+01, 9.00E+01, 1.00E+02, 2.00E+02, 3.00E+02, 4.00E+02, 5.00E+02, 6.00E+02, 7.00E+02, 8.00E+02, 9.00E+02, 1.00E+03, 2.00E+03, 3.00E+03, 4.00E+03, 5.00E+03, 6.00E+03, 7.00E+03, 8.00E+03, 9.00E+03, 1.00E+04, 2.00E+04, 3.00E+04, 4.00E+04, 5.00E+04, 6.00E+04, 7.00E+04, 8.00E+04, 9.00E+04, 1.00E+05, 2.00E+05, 3.00E+05, 4.00E+05, 5.00E+05, 6.00E+05, 7.00E+05, 8.00E+05, 9.00E+05, 1.00E+06, 2.00E+06, 3.00E+06, 4.00E+06, 5.00E+06, 6.00E+06, 7.00E+06, 8.00E+06, 9.00E+06, 1.00E+07, 2.00E+07, 3.00E+07, 4.00E+07, 5.00E+07, 6.00E+07, 7.00E+07, 8.00E+07, 9.00E+07, 1.00E+08],
                      [1.1646000E-25, 2.3577900E-25, 3.2591000E-25, 4.2992800E-25, 5.1960300E-25, 6.0276100E-25, 6.9943600E-25, 7.9513400E-25, 8.8465600E-25, 9.8417700E-25, 1.8598000E-24, 2.7048100E-24, 3.5377700E-24, 4.3460900E-24, 5.1305600E-24, 5.9122000E-24, 6.6789000E-24, 7.4485800E-24, 8.2041400E-24, 1.5489600E-23, 2.2429900E-23, 2.9145600E-23, 3.5689800E-23, 4.2108100E-23, 4.8413600E-23, 5.4623000E-23, 6.0744800E-23, 6.6796500E-23, 1.2425000E-22, 1.7800600E-22, 2.2931200E-22, 2.7876700E-22, 3.2674600E-22, 3.7347900E-22, 4.1915100E-22, 4.6388000E-22, 5.0778500E-22, 9.1413800E-22, 1.2818800E-21, 1.6250500E-21, 1.9503500E-21, 2.2616900E-21, 2.5616300E-21, 2.8519800E-21, 3.1341000E-21, 3.4090000E-21, 5.8872100E-21, 8.0639000E-21, 1.0061900E-20, 1.1936700E-20, 1.3720100E-20, 1.5432200E-20, 1.7087500E-20, 1.8696500E-20, 2.0267200E-20, 3.4876300E-20, 4.9101500E-20, 6.4106500E-20, 8.0522100E-20, 9.8770900E-20, 1.1916500E-19, 1.4194600E-19, 1.6731000E-19, 1.9541700E-19, 6.4807800E-19, 1.4392500E-18, 2.5696300E-18, 4.0202100E-18, 5.7660900E-18, 7.7816300E-18, 1.0042400E-17, 1.2526100E-17, 1.5212400E-17, 5.0039600E-17, 9.3437200E-17, 1.4117900E-16, 1.9139500E-16, 2.4308300E-16, 2.9569500E-16, 3.4889000E-16, 4.0244400E-16, 4.5620900E-16, 9.9233700E-16, 1.5145500E-15, 2.0213000E-15, 2.5142100E-15, 2.9949700E-15, 3.4649900E-15, 3.9254300E-15, 4.3772200E-15, 4.8211600E-15, 8.9373200E-15, 1.2653000E-14, 1.6103700E-14, 1.9357200E-14, 2.2454400E-14, 2.5422400E-14, 2.8280900E-14, 3.1044300E-14, 3.3724100E-14, 5.7326900E-14, 7.7294300E-14, 7.6002200E-14, 9.1379800E-14, 1.0571600E-13, 1.1920500E-13, 1.3198300E-13, 1.4415300E-13, 1.5579500E-13, 2.5273100E-13, 3.2863300E-13, 3.9253800E-13, 4.4837300E-13, 4.9828800E-13, 5.4362100E-13, 5.8527100E-13, 6.2387900E-13, 6.5992000E-13, 9.3304800E-13, 1.1213700E-12, 1.2668600E-12, 1.3858400E-12, 1.4866100E-12, 1.5740100E-12, 1.6511600E-12, 1.7201700E-12, 1.7825500E-12, 2.2012700E-12, 2.4448000E-12, 2.6126800E-12, 2.7385600E-12, 2.8378900E-12, 2.9190400E-12, 2.9870500E-12, 3.0451400E-12, 3.0955300E-12, 3.3867200E-12, 3.5229800E-12, 3.6045600E-12, 3.6596100E-12, 3.6995600E-12, 3.7300100E-12, 3.7540500E-12, 3.7735600E-12, 3.7897400E-12, 3.8701300E-12, 3.9004900E-12, 3.9165600E-12, 3.9265200E-12, 3.9333100E-12, 3.9382300E-12, 3.9419700E-12, 3.9449100E-12, 3.9472700E-12, 3.9581100E-12, 3.9618000E-12, 3.9636600E-12, 3.9647800E-12, 3.9655200E-12, 3.9660600E-12, 3.9664600E-12, 3.9667800E-12, 3.9670300E-12, 3.9681600E-12, 3.9685400E-12, 3.9687300E-12, 3.9688400E-12, 3.9689200E-12, 3.9689700E-12, 3.9690100E-12, 3.9690400E-12, 3.9690700E-12, 3.9691800E-12, 3.9692200E-12, 3.9692400E-12, 3.9692500E-12, 3.9692600E-12, 3.9692600E-12, 3.9692700E-12, 3.9692700E-12, 3.9692700E-12, 3.9692800E-12, 3.9692900E-12, 3.9692900E-12, 3.9692900E-12, 3.9692900E-12, 3.9692900E-12, 3.9692900E-12, 3.9692900E-12, 3.9692900E-12, 3.9692900E-12, 3.9692900E-12, 3.9692900E-12, 3.9692900E-12, 3.9692900E-12, 3.9692900E-12, 3.9692900E-12, 3.9692900E-12, 3.9692900E-12]
                      ]) # 2d-array of (pressure, rate constants) for component 1 at T indicated above

    k_P_2 = np.array([[1.00E-14, 2.00E-14, 3.00E-14, 4.00E-14, 5.00E-14, 6.00E-14, 7.00E-14, 8.00E-14, 9.00E-14, 1.00E-13, 2.00E-13, 3.00E-13, 4.00E-13, 5.00E-13, 6.00E-13, 7.00E-13, 8.00E-13, 9.00E-13, 1.00E-12, 2.00E-12, 3.00E-12, 4.00E-12, 5.00E-12, 6.00E-12, 7.00E-12, 8.00E-12, 9.00E-12, 1.00E-11, 2.00E-11, 3.00E-11, 4.00E-11, 5.00E-11, 6.00E-11, 7.00E-11, 8.00E-11, 9.00E-11, 1.00E-10, 2.00E-10, 3.00E-10, 4.00E-10, 5.00E-10, 6.00E-10, 7.00E-10, 8.00E-10, 9.00E-10, 1.00E-09, 2.00E-09, 3.00E-09, 4.00E-09, 5.00E-09, 6.00E-09, 7.00E-09, 8.00E-09, 9.00E-09, 1.00E-08, 2.00E-08, 3.00E-08, 4.00E-08, 5.00E-08, 6.00E-08, 7.00E-08, 8.00E-08, 9.00E-08, 1.00E-07, 2.00E-07, 3.00E-07, 4.00E-07, 5.00E-07, 6.00E-07, 7.00E-07, 8.00E-07, 9.00E-07, 1.00E-06, 2.00E-06, 3.00E-06, 4.00E-06, 5.00E-06, 6.00E-06, 7.00E-06, 8.00E-06, 9.00E-06, 1.00E-05, 2.00E-05, 3.00E-05, 4.00E-05, 5.00E-05, 6.00E-05, 7.00E-05, 8.00E-05, 9.00E-05, 1.00E-04, 2.00E-04, 3.00E-04, 4.00E-04, 5.00E-04, 6.00E-04, 7.00E-04, 8.00E-04, 9.00E-04, 1.00E-03, 2.00E-03, 3.00E-03, 4.00E-03, 5.00E-03, 6.00E-03, 7.00E-03, 8.00E-03, 9.00E-03, 1.00E-02, 2.00E-02, 3.00E-02, 4.00E-02, 5.00E-02, 6.00E-02, 7.00E-02, 8.00E-02, 9.00E-02, 1.00E-01, 2.00E-01, 3.00E-01, 4.00E-01, 5.00E-01, 6.00E-01, 7.00E-01, 8.00E-01, 9.00E-01, 1.00E+00, 2.00E+00, 3.00E+00, 4.00E+00, 5.00E+00, 6.00E+00, 7.00E+00, 8.00E+00, 9.00E+00, 1.00E+01, 2.00E+01, 3.00E+01, 4.00E+01, 5.00E+01, 6.00E+01, 7.00E+01, 8.00E+01, 9.00E+01, 1.00E+02, 2.00E+02, 3.00E+02, 4.00E+02, 5.00E+02, 6.00E+02, 7.00E+02, 8.00E+02, 9.00E+02, 1.00E+03, 2.00E+03, 3.00E+03, 4.00E+03, 5.00E+03, 6.00E+03, 7.00E+03, 8.00E+03, 9.00E+03, 1.00E+04, 2.00E+04, 3.00E+04, 4.00E+04, 5.00E+04, 6.00E+04, 7.00E+04, 8.00E+04, 9.00E+04, 1.00E+05, 2.00E+05, 3.00E+05, 4.00E+05, 5.00E+05, 6.00E+05, 7.00E+05, 8.00E+05, 9.00E+05, 1.00E+06, 2.00E+06, 3.00E+06, 4.00E+06, 5.00E+06, 6.00E+06, 7.00E+06, 8.00E+06, 9.00E+06, 1.00E+07, 2.00E+07, 3.00E+07, 4.00E+07, 5.00E+07, 6.00E+07, 7.00E+07, 8.00E+07, 9.00E+07, 1.00E+08],
                      [1.86732000E-24, 3.53096000E-24, 5.13717000E-24, 6.68872000E-24, 8.20465000E-24, 9.69892000E-24, 1.11720000E-23, 1.26315000E-23, 1.40632000E-23, 1.54863000E-23, 2.91402000E-23, 4.21139000E-23, 5.46122000E-23, 6.67973000E-23, 7.86929000E-23, 9.03585000E-23, 1.01822000E-22, 1.13118000E-22, 1.24248000E-22, 2.29317000E-22, 3.26755000E-22, 4.19164000E-22, 5.07799000E-22, 5.93425000E-22, 6.76557000E-22, 7.57550000E-22, 8.36682000E-22, 9.14172000E-22, 1.62510000E-21, 2.26175000E-21, 2.85206000E-21, 3.40909000E-21, 3.94039000E-21, 4.45085000E-21, 4.94392000E-21, 5.42211000E-21, 5.88737000E-21, 1.00622000E-20, 1.37204000E-20, 1.70880000E-20, 2.02677000E-20, 2.33191000E-20, 2.62810000E-20, 2.91814000E-20, 3.20413000E-20, 3.48773000E-20, 6.41086000E-20, 9.87750000E-20, 1.41953000E-19, 1.95427000E-19, 2.60380000E-19, 3.37634000E-19, 4.27766000E-19, 5.31175000E-19, 6.48122000E-19, 2.56981000E-18, 5.76649000E-18, 1.00431000E-17, 1.52134000E-17, 2.11235000E-17, 2.76505000E-17, 3.46960000E-17, 4.21812000E-17, 5.00424000E-17, 1.41186000E-16, 2.43094000E-16, 3.48904000E-16, 4.56228000E-16, 5.64013000E-16, 6.71739000E-16, 7.79131000E-16, 8.86038000E-16, 9.92374000E-16, 2.02137000E-15, 2.99507000E-15, 3.92555000E-15, 4.82131000E-15, 5.68820000E-15, 6.53043000E-15, 7.35118000E-15, 8.15291000E-15, 8.93759000E-15, 1.61042000E-14, 2.24550000E-14, 2.82816000E-14, 3.37250000E-14, 3.88683000E-14, 4.37662000E-14, 4.84574000E-14, 5.29708000E-14, 5.73284000E-14, 7.60045000E-14, 1.05719000E-13, 1.31986000E-13, 1.55799000E-13, 1.77735000E-13, 1.98168000E-13, 2.17360000E-13, 2.35501000E-13, 2.52737000E-13, 3.92547000E-13, 4.98298000E-13, 5.85282000E-13, 6.59932000E-13, 7.25699000E-13, 7.84688000E-13, 8.38299000E-13, 8.87516000E-13, 9.33063000E-13, 1.26688000E-12, 1.48663000E-12, 1.65118000E-12, 1.78257000E-12, 1.89170000E-12, 1.98481000E-12, 2.06581000E-12, 2.13735000E-12, 2.20129000E-12, 2.61270000E-12, 2.83791000E-12, 2.98706000E-12, 3.09554000E-12, 3.17906000E-12, 3.24590000E-12, 3.30091000E-12, 3.34717000E-12, 3.38674000E-12, 3.60456000E-12, 3.69957000E-12, 3.75406000E-12, 3.78974000E-12, 3.81504000E-12, 3.83397000E-12, 3.84868000E-12, 3.86047000E-12, 3.87013000E-12, 3.91656000E-12, 3.93331000E-12, 3.94197000E-12, 3.94727000E-12, 3.95085000E-12, 3.95342000E-12, 3.95537000E-12, 3.95689000E-12, 3.95811000E-12, 3.96366000E-12, 3.96552000E-12, 3.96646000E-12, 3.96703000E-12, 3.96740000E-12, 3.96767000E-12, 3.96788000E-12, 3.96803000E-12, 3.96816000E-12, 3.96873000E-12, 3.96892000E-12, 3.96901000E-12, 3.96907000E-12, 3.96910000E-12, 3.96913000E-12, 3.96915000E-12, 3.96917000E-12, 3.96918000E-12, 3.96924000E-12, 3.96926000E-12, 3.96927000E-12, 3.96927000E-12, 3.96928000E-12, 3.96928000E-12, 3.96928000E-12, 3.96928000E-12, 3.96928000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12, 3.96929000E-12]
                      ]) # 2d-array of (pressure, rate constants) for component 2 at T indicated above

    k_P = [k_P_1, k_P_2]
    k_LMR = LMR_R(T,P,X,eig_0,k_P) # calculate LMR,R rate constant

    # NMR,R implementation
    delta_e_d = np.array([70., 400.]) * (T / 298.15) ** 0.65  # averaged energy transferred per downward collision
    Z = [1e-10, 1e-10] # collision frequency
    Fe = 1.8116761923287874 # Fe describing the energy dependence of the density of states near the lowest decomposition threshold, dimensionless
    f_list = activity_coefficient(T, X, Z, delta_e_d, Fe) # solve for activity coefficients

    eig_0 = [4.005782458300001e-24, 8.009631798840002e-23] # rate coefficients in the low-pressure limit for each collider
    k_NMR = NMR_R(T,P,X,eig_0,f_list,k_P) # calculate NMR,R rate constant
