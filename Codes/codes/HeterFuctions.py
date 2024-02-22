
#######################################################################################################################
#######################################################################################################################
# 1.**Import modules**

import pandas as pd
import numpy as np
import os, math
from math import *
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.special import comb
from dtaidistance import dtw_ndim
import scipy.io as scio

from joblib import Parallel, delayed
import copy
from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore")





#######################################################################################################################
#######################################################################################################################
## 2.1.**Vehicular dynamics** 

def Get_beta(sigma, lr, lf):
    '''
        Calculate the 'β', which is the angle. It's
        the angle difference between the velocity of the
        c.g. of the vehicle and the longitudinal axis of
        the vehicle.
        
        ------------------------------------------------
        @input:
            sigma : the steering angle of the front wheels, (unit: rad)
            lr    : the distance between rear wheel axis and the c.g. of the vehicle
            lf    : the distance between front wheel axis and the c.g. of the vehicle
        
        @output:
            beta  : (unit: rad) 
    '''
    #*********************************Calculate the angle
    beta = atan(lr*tan(sigma) / (lr+lf))
    
    #**********************************Constrain the domain
    if beta > 60 * np.pi / 180:
        beta = 60 * np.pi / 180
    elif beta < -60 * np.pi / 180:
        beta = -60 * np.pi / 180
    return beta   
    

def Get_phi(V, sigma, beta, lr, lf, dt, phi_last):
    '''
        Calculate the 'φ', which is vehicle's heading angle
    '''
    #*********************************Calculate the angle
    phi_new = radians((V*cos(beta)*tan(sigma) / (lf+lr)) * dt) + phi_last
    
    #**********************************Constrain the domain
    if phi_new > 90 *  np.pi / 180:
        phi_new = 90 *np.pi / 180
    elif phi_new < -90 * np.pi / 180:
        phi_new = -90 * np.pi / 180
    
    return phi_new
    

def VehicleStatus(Va, V_new, x_last, y_last, vx_last, vy_last, phi, sigma, dt):
    '''
        Calculate vehicle's status at current time
    '''
    
    vx = V_new * cos(phi+sigma)
    vy = V_new * sin(phi+sigma)
    
    ax = Va * cos(phi+sigma)
    ay = Va * sin(phi+sigma)
    
    x = x_last + vx*dt + 0.5*ax*dt**2
    y = y_last + vy*dt + 0.5*ay*dt**2

    return vx, vy, ax, ay, x, y


def GetVehicleStatus(x_last, y_last, vx_last, vy_last, phi, sigma, dt, Va, V_new):
    
    vy = vx_new * tan(phi+sigma)
    ay = (vy - vy_last) / dt
    y = y_last + vy*dt + 0.5*ay*dt**2
    
    vx = vx_new
    ax = ax
    x = x_last + vx_new*dt + 0.5*ax*dt**2
    

    return vx, vy, ax, ay, x, y


def Get_sigma(v_sigma_last, a_sigma_d, v_sigma_d, dt, sigma_last, alpha, lambda1, lambda2):
    '''
        Calculate the 'σ', which is the vehicle's 
        front steering angle in next time step
        
        ---------------------------------------------------------------
        @input:
            v_sigma_last: the steering speed of front wheels in last time step, unit: rad/s
            a_sigma_d   : the desired steering acceleration of front wheels, (unit: rad/s^2)
            v_sigma_d   : the desired steering speed of front wheels, (unit: rad/s)
            dt          : the time step, (unit: s)
            sigma_last  : the steering angle of front wheels in last time step, (unit: rad)
            alpha       : the relative lateral distance angle, (unit: rad)
            lambda1     : the 'σ' sensitivity coefficient, (unit: --)
            lambda2     : the 'Θ' sensitivity coefficient, (unit: --)
            
        @output:
            sigma (σ)   : (unit: rad)
            v_sigma     : the steering speed of front wheel, (unit: rad/s)
    '''
    #*********************************Calculate the elements
#     a_sigma = a_sigma_d * (1 - (v_sigma_last/v_sigma_d)**4 - exp(sigma_last) - Lambda*tan(-alpha))
    a_sigma = a_sigma_d * (1 - (v_sigma_last/v_sigma_d)**4 - exp(lambda1*sigma_last) - lambda2*tan(-alpha))      
    v_sigma = a_sigma * dt + v_sigma_last
    sigma = 0.5*a_sigma*dt**2 + v_sigma*dt + sigma_last 
    
    #**********************************Constrain the domain
    if sigma > 75 *  np.pi / 180:
        sigma = 75 *np.pi / 180
    elif sigma < -75 * np.pi / 180:
        sigma = -75 * np.pi / 180
        
    return sigma, v_sigma


def Get_alpha(x_e, y_e, xt, yt):
    '''
        Calculate the 'Θ'， which is the angle between the vector, [1,0] 
        and the vector from the c.g. of theagent vehicle to its expected 
        position [x_e, y_e]
        
        ---------------------------------
        @input:
            [x_e, y_e]: ... 
        
        @output:
            Θ : (unit: rad)
        
    '''
#     alpha = atan(abs(y_e-yt)/(x_e-xt))
    alpha = atan((y_e-yt)/(x_e-xt))
    return alpha


def IDM(T, vd, s, a, b, s0, dt, d_v, V_last, v_sigma, v_sigma_d):
    '''
        The two-dimensional intelligent model
    '''
    s_start = max(0, s0 + V_last*T + V_last*d_v/(2*sqrt(a*b)))
    Va = a * (1 - (V_last/vd)**4 - (s_start/s)**2 - (v_sigma/v_sigma_d)**2)
    V_new = Va*dt + V_last
    return Va, V_new
    
    


#######################################################################################################################
#######################################################################################################################
## 1.2.**Potential model＋MPC**

def Gamma(a, b, d, amplitude):
    '''
        Generate the 'gamma function' shape
    '''    
    #*********************Strength of the potential field
    z = amplitude * d**(a-1) * np.exp(-b*d) * b**a / math.factorial(a-1)
    
    return z


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, coeff_shape, nTimes=1e4):
    """
        Generate Bezier Curve based on Two Given Points
        -----------------------------------------------
        @input:
            points: start point,[X1,Y1]; end point, [X2,Y2].               
            coeff_shape: the shape control coefficients, [k1,k2,k3,k4]
            
        @output:
            return: [[x1,y1], [x2,y2]].
    """
    p0 = points[0]  # start point
    pn = points[1]  # end point
    k1,k2,k3,k4 = coeff_shape
    
    ## add two control points
    p1 = [k1*(p0[0]+pn[0]), k2*(p0[1]+pn[1])]
    p2 = [k3*(p0[0]+pn[0]), k4*(p0[1]+pn[1])]
    points = np.array([p0, p1, p2, pn])
    
    ## Bezier curve 
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    t = np.linspace(0.0, 1.0, nTimes)
    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    
    return xvals, yvals


def get_line(X, Y):
    '''
        Get the slop and intercept based
        given two points
        --------------------------------
        
        @input:
                X: the x-coordinates of these points, [x1, x2, x3, ...]
                Y: the y-coordinates of these points, [y1, y2, y3, ...]
        
        @output:
                k: the slop of the line
                b: the intercept of the line
                x: the new x-coordinates
                y: the new y-coordinates
    '''
    #*******************Calculate the slop and intercept
    x1, x2 = X
    y1, y2 = Y
    k = (y2-y1) / (x2-x1)
    b = y1 - k*x1
    
    #*******************Calculate the slop and intercept
    x = np.arange(x1, x2, 0.1)
    y = k*x + b
    return k, b, x, y


def Potential_VBP(xvals, yvals, xt, yt, α, β):
    '''
        Generate the potential of the 'virtual boundary'
        
        ----------------------------------
        @input:
            xvals, yvals: coordinates of virtual boundary
            xt, yt      : coordinate of the agent vehicle
            α           : the maximum value of the potential field
            β           : the raise/fall rate of the potential field
    '''
    #*********************Boundaries the potential field   
    xb_1, xb_2, yb_1, yb_2 = [xvals.min(), xvals.max(), yvals.min(), yvals.max()]
   
    #*********************Find the closest virtual boundary points
    distance = np.sqrt((xvals-xt)**2 + (yvals-yt)**2)
    idx = distance.argmin()
    p_min = xvals[idx], yvals[idx]
    sign = p_min[0]-xt, p_min[1]-yt
    
    #*********************Strength of the potential field
    if sign[0]>=0 and sign[1]<=0:
        dmin = distance.min()
        if dmin<=1e-2:
            U = α
        else:
            U = α * np.exp(-β*dmin)
    else:
        U = 0
    #*********************Remove abnormal value
    if yt < yb_1:
        U = 0
        
    return U


def Potential_Vehicle(x,y,xt,yt,l,w,V,k1,k2,k3):
    '''
        Calculte the strength of the vehicle's 
        potential field
        -------------------------------------
        @input:
            x: the x-coordinate of the agent vehicle
            y: the y-coordinate of the agent vehicle
            xt: the x-coordinate of the surrounding vehicle
            yt: the y-coordinate of the surrounding vehicle
            l: the length of the surrounding vehicle
            w: the width of the surrounding vehicle
            V: the relative speed
            k1: amplitude coefficient
            k2: range coefficient
            k3: shape coefficient
        
    '''
    U = k1*np.exp( -(k2*((x-xt)**2/(l**2)+(y-yt)**2)/(w**2))**k3 / (2*V) ) 
    return U


class Potential:
    '''
        Calculate the boundary generated by the road topology
        -----------------------------------------------------
        
        @input:
            X     : the meshigrid x
            Y     : the meshigrid x
            points: the terminal points of a line, [[x1,x2], [y1,y2]]
            P     : the maximum value of the potential
            tag   : the direction of the potential field, -1: represents the 'left' or 'down'; 1: represents the 'right' or 'up'
                    0: represents both sides
    '''
    @classmethod
    def StraightBoundary(self, X, Y, points, P, σ, tag):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~Calculate the potential
        k,b,xx,yy = get_line(points[0], points[1])
        yt = k*X + b
        dist = abs(k*X-Y+b)/sqrt(k**2+1)
        U = P * np.exp( -(dist) / (2*σ**2) )
        
        #~~~~~~~~~~~~~~~~~~~~~~~~Remove non-potential
        if tag==-1:
            x_index, y_index = np.where(Y<b)
            U[x_index, y_index] = 0
        elif tag==1:
            x_index, y_index = np.where((Y>b) | (X<150))
            U[x_index, y_index] = 0
            
        return U
    
    
    @classmethod
    def RampBoundary(self, X, Y, points, P, σ, tag):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~Calculate the potential
        k,b,xx,yy = get_line(points[0], points[1])
        yt = k*X + b
        dist = abs(k*X-Y+b)/sqrt(k**2+1)
        U = P * np.exp( -(dist) / (2*σ**2) )
        
        #~~~~~~~~~~~~~~~~~~~~~~~~Judge the position whether the points are on the dowm/up directio of the boundary
        sign = np.zeros(U.shape)
        for i in range(sign.shape[0]):
            for j in range(sign.shape[1]):
                if Y[i,j] > X[i,j]*k+b:
                    sign[i,j] = 1
                elif Y[i,j] < X[i,j]*k+b:
                    sign[i,j] = -1
                    
       #~~~~~~~~~~~~~~~~~~~~~~~~Remove non-potential 
        if tag==-1:
            x_index, y_index = np.where(sign==1)
            U[x_index, y_index] = 0
        elif tag==1:
            x_index, y_index = np.where((sign==-1) | (X<=150))
            U[x_index, y_index] = 0
            
        return U
    
    
    @classmethod
    def LaneMarkings(self, X, Y, points, P, σ, tag):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~Calculate the potential
        k,b,xx,yy = get_line(points[0], points[1])
        yt = k*X + b
        dist = abs(k*X-Y+b)/sqrt(k**2+1)
        U = P * np.exp( -(dist) / (2*σ**2))
        
        if tag==1:
            x_index, y_index = np.where(X>=150)
            U[x_index, y_index] = 0
 
        return U
    
    
    @classmethod
    def VirtualBoundary(self, X, Y, xvals, yvals, X0, Y0, σ, α, β):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~Calculate the potential
        U_vbp = np.zeros(X.shape)
        for i in range(X0.shape[0]):
            for j in range(Y0.shape[0]):
                U_vbp[j,i] = Potential_VBP(xvals, yvals, X0[i], Y0[j], α, β)
        
        return U_vbp


def Gradient(U, phi):
    '''
        Calculate the potential gradient 
    '''
    G = np.gradient(U)[0] * cos(phi) + np.gradient(U)[1] * sin(phi)
    
    return G



#######################################################################################################################
## --- MPC Controller


class MPC:
    def __init__(self, dt, horizon, tracks_planned):
        self.R = np.diag([0.01, 0.01])          # input cost matrix
        self.Rd = np.diag([0.01, 1.0])          # input difference cost matrix
        self.Q = np.diag([1.0, 1.0])            # state cost matrix
        self.Qf = self.Q                        # state final matrix
        
        self.dt = dt                            # simulation time step
        self.horizon = horizon                  # number of points is used to optimize per iteration
        self.tracks_planned = tracks_planned    # the planned tracks                       

    def move(self, my_car, accelerate, delta):
        '''
        Move car.
        '''
        x_dot = my_car.v*np.cos(my_car.psi+my_car.beta)
        y_dot = my_car.v*np.sin(my_car.psi+my_car.beta) 
        v_dot = accelerate
        psi_dot = my_car.v * np.cos(my_car.beta) * np.tan(delta)/self.L
        beta_dot = np.arctan(0.5*np.tan(delta))
        return np.array([[x_dot, y_dot, v_dot, psi_dot, beta_dot]]).reshape(5,1)

    def update_my_car(self, my_car, state_dot):
        '''
        Update car's state.
        '''
        # self.u_k = command
        # self.z_k = state
        state = np.array([[my_car.x, my_car.y, my_car.v, my_car.psi, my_car.delta]]).reshape(5,1)
        state = state + self.dt*state_dot
        my_car['x'] = state[0,0]
        my_car['y'] = state[1,0]
        my_car['v'] = state[2,0]
        my_car['psi'] = state[3,0]
        my_car['beta'] = state[4,0]
        return my_car
            
    def mpc_cost(self, u_k, my_car, points):
        '''
        Calculate cost.
        '''
        mpc_car = copy.copy(my_car)
        u_k = u_k.reshape(self.horiz, 2).T
        z_k = np.zeros((2, self.horiz+1))
        
        desired_state = points.T
        cost = 0.0
        for i in range(self.horiz):
            state_dot = self.move(mpc_car, u_k[0,i], u_k[1,i])
            mpc_car =  self.update_my_car(mpc_car, state_dot)
        
            z_k[:,i] = [mpc_car.x, mpc_car.y]
            cost += np.sum(self.R@(u_k[:,i]**2))
            cost += np.sum(self.Q@((desired_state[:,i]-z_k[:,i])**2))
            if i < (self.horiz-1):     
                cost += np.sum(self.Rd@((u_k[:,i+1] - u_k[:,i])**2))
        return cost

    def optimize(self, my_car, points):
        '''
        Optimization.
        '''
        self.horiz = points.shape[0]
        # bnd = [(-5, 5),(np.deg2rad(-60), np.deg2rad(60))]*self.horiz
        max_angle = 75
        bnd = [(-11.5, 11.5),(-1.066, 1.066)]*self.horiz
        result = minimize(self.mpc_cost, args=(my_car, points), x0 = np.zeros((2*self.horiz)), method='slsqp', bounds = bnd)
        return result.x[0],  result.x[1]

    def update_state(self, my_car, points):
        '''
        Update car's state...
        '''
        ## optimize the control
        control = self.optimize(my_car, points)
        
        ## move for a period of time
        accelerate, delta = control
        dx, dy, dv, dpsi, dbeta = self.move(my_car, accelerate, delta) * self.dt
        
        ## calculate the new state
        old_state = np.array([my_car.x, my_car.y, my_car.v, my_car.psi, my_car.beta]).reshape(5,1)
        new_state = list(old_state + np.array([dx, dy, dv, dpsi, dbeta]).reshape(5,1))
        return [accelerate, delta], new_state
    
    def estimate_track(self, car):
        '''
        Estimate car's track (psi, beta, delta,...).
        '''
        ## get the length of the car
        self.L = car.length.iloc[0]
        
        
        ## create the track of my car
        num = car.shape[0]
        track = pd.DataFrame(np.zeros((num,8)), columns=['frame','x', 'y', 'v', 'psi', 'beta','delta', 'a'])
        if 'Velocity' not in car.columns:
            track.iloc[0] = car.iloc[0][['frame', 'x', 'y']].tolist() + [np.linalg.norm(car.iloc[0][['vx','vy']]),0,0,0,0]
        else:
            track.iloc[0] = car.iloc[0][['frame', 'x', 'y','Velocity']].tolist() + [0,0,0,0]

        ## loop for estimation
        for i in range(1,num):
            ## prepare my_car and future points
            my_car = track.iloc[i-1]
            if i <= num - self.horizon:
                points = car[['x','y']].values[i:i+self.horizon]
            else:
                points = car[['x','y']].values[i:]
        
            ## optimize control and get new state
            control, new_state = self.update_state(my_car, points)

            ## update information of track
            track.iloc[i,0] = car.frame.iloc[i]
            track.iloc[i,1:6] = new_state
            track.iloc[i-1,6] = control[1]
            track.iloc[i-1,7] = control[0]
        
        ## add car's id
        track['id'] = car.id.iloc[0]

        return track
    
    def main(self,):
        '''
        Optimize the planned tracks 
        with parallel calculation.
        '''
        ## get car's track 
        # vehicles = [self.tracks_planned[self.tracks_planned.id==id] for id in self.tracks_planned.id.unique()]

        ## Create the parallel calculation    
        results = Parallel(n_jobs=-1)(delayed(self.estimate_track)(veh) for veh in [self.tracks_planned])

        ## concat dataframe
        tracks_mpc = pd.concat(results)
        
        return tracks_mpc


class Error:
    '''
        Error analysis: MAE, MSE, RMSE
    '''
    @classmethod
    def MAE(self,x_real, y_real, x_sim, y_sim):
        x_real = x_real.tolist()
        y_real = y_real.tolist()
        distance = []
        for i in range(len(x_real)):
            dist = sqrt((x_real[i]-x_sim[i])**2 + (y_real[i]-y_sim[i])**2)
            distance.append(dist)
        return np.mean(distance)
    
    @classmethod
    def MSE(self,x_real, y_real, x_sim, y_sim):
        x_real = x_real.tolist()
        y_real = y_real.tolist()
        distance = []
        for i in range(len(x_real)):
            dist = sqrt((x_real[i]-x_sim[i])**2 + (y_real[i]-y_sim[i])**2)
            distance.append(dist**2)
        return np.mean(distance)
    
    @classmethod
    def RMSE(self,x_real, y_real, x_sim, y_sim):
        
        return np.sqrt(Error.MSE(x_real, y_real, x_sim, y_sim))
    
    @classmethod
    def MAPE(self,x_real, y_real, x_sim, y_sim):
        x_real = x_real.tolist()
        y_real = y_real.tolist()
        distance = []
        for i in range(len(x_real)):
            p1 = abs(x_real[i]-x_sim[i])/x_real[i]
            p2 = abs(y_real[i]-y_sim[i])/y_real[i]
            distance.append((p1+p2))
        return np.mean(distance)
    
    @classmethod
    def Analysis(self,x_real, y_real, x_sim, y_sim):
        '''
            Calculate all error indexes
        '''
        dtw = dtw_ndim.distance(np.array([x_sim, y_sim]).T, np.array([x_real, y_real]).T)
        mae = self.MAE(x_real, y_real, x_sim, y_sim)
        mse = self.MSE(x_real, y_real, x_sim, y_sim)
        rmse = self.RMSE(x_real, y_real, x_sim, y_sim)
        mape = self.MAPE(x_real, y_real, x_sim, y_sim)
        return dtw, mae, mse, rmse, mape





#######################################################################################################################
#######################################################################################################################
## 1.3.**Visulization function**

def visulize_raw_trajectory(dataframe):
    '''
        Plot all vehicle's trajectories in the dataframe
    '''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~Plot trajectories
    group = dataframe.groupby('id')
    vId = dataframe.id.unique()
    fig, ax = plt.figure(figsize=(10,6)), plt.axes()
    figs = [plt.plot(group.get_group(id).x, group.get_group(id).y) for id in vId]
    plt.xlabel('Longitudinal position (m)',fontsize=14)
    plt.ylabel('Lateral position (m)',fontsize=14)
    
    
def get_vehicle_size(num):
    '''
        Calculate vheicle's size for 'mirror-traffic' data
    '''
    if num==3:
        l,w = 4.05, 1.8
    elif num==4:
        l,w = 1.9, 0.75
    elif num==6:
        l,w = 14, 2.59
    elif num==8:
        l,w = 10, 2.25
    
    return l,w
    

def data_columns_selection(dataframe):
    '''
        Select the specific columns: 'highD', 'exiD', 'Mirror-traffic'
        ---------------------------
    '''
    if dataframe.shape[1]==25:
        dataframe.rename(columns=({'width':'length', 'height':'width'}), inplace=True) 
        
    elif dataframe.shape[1]==29:
        dataframe.rename(columns=({'trackId':'id', 'xCenter':'x', 'yCenter':'y'}), inplace=True)
        
    elif dataframe.shape[1]==16:
        dataframe = dataframe.drop('width', 1)
        dataframe['size'] = dataframe['classId'].apply(get_vehicle_size)
        dataframe['length'] = dataframe['size'].apply(lambda x: x[0])
        dataframe['width'] = dataframe['size'].apply(lambda x: x[1])
        dataframe.rename(columns=({'frameId':'frame','trackId':'id', 'localY':'x', 'localX':'y', 'xVelocity':'yVelocity', 
                                   'yVelocity':'xVelocity', 'xAcceleration':'yAcceleration', 'yAcceleration':'xAcceleration'}), inplace=True) 
        
    # data = dataframe[['frame','id','x','y','xVelocity','yVelocity','xAcceleration','yAcceleration','laneId','length','width']]
    data = dataframe
    return data


def plot_topology():
    '''
        Plot these lane markings for 'Mirror-traffic'
    '''
    # Add Road Boundaries------------------------------------------------------------------------------------------------
    plt.plot([0, 250], [0.5, 0.5], ls="-",c="k", lw=3)
    plt.plot([0, 250], [3.7, 3.7],ls="--",c="k", lw=1.5) # add lane boundary
    plt.plot([0, 250], [7.6, 7.6],ls="--",c="k", lw=1.5) # add lane boundary
    plt.plot([0, 250], [7.6, 7.6],ls="--",c="k", lw=1.5) # add lane boundary
    plt.plot([50, 230], [7.6, 13.5], ls="--",c="k", lw=1.5, label='lane marks')

    plt.plot([0, 230], [8.5, 16.5], ls="-",c="k", lw=3)
    plt.plot([150, 250], [7.6, 11], ls="-",c="k", lw=3)
    plt.plot([150, 250], [7.6, 7.6], ls="-",c="k", lw=3, label='boundary')
    
    
def filter_lane_keep_data(full_data):
    '''
        Filter the lane-keep data 
        from the full data
    '''
    vids = full_data.id.unique()

    ## filter lane-keep
    vehicles = []
    for id in vids:
        veh = full_data[full_data['id']==id]
        if veh.laneId.unique().shape[0] == 1:
            vehicles.append(veh)

    ## concat these sub-dataframe
    vehicles = pd.concat(vehicles)
    
    return vehicles


def filter_lane_change_data(full_data):
    '''
        Filter the lane-keep data 
        from the full data
    '''
    vids = full_data.id.unique()

    ## filter lane-keep
    vehicles = []
    for id in vids:
        veh = full_data[full_data['id']==id]
        if veh.laneId.unique().shape[0] > 1:
            vehicles.append(veh)

    ## concat these sub-dataframe
    vehicles = pd.concat(vehicles)
    
    return vehicles






#######################################################################################################################
#######################################################################################################################
## 2.2.**Generate potential fields**

class Potential:
    '''
        Calculate the boundary generated by the road topology
        -----------------------------------------------------
        
        @input:
            X     : the meshigrid x
            Y     : the meshigrid x
            points: the terminal points of a line, [[x1,x2], [y1,y2]]
            P     : the maximum value of the potential
            tag   : the direction of the potential field, -1: represents the 'left' or 'down'; 1: represents the 'right' or 'up'
                    0: represents both sides
    '''
    @classmethod
    def StraightBoundary(self, X, Y, points, P, σ, tag):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~Calculate the potential
        k,b,xx,yy = get_line(points[0], points[1])
        yt = k*X + b
        dist = abs(k*X-Y+b)/sqrt(k**2+1)
        U = P * np.exp( -(dist) / (2*σ**2) )
        
        #~~~~~~~~~~~~~~~~~~~~~~~~Remove non-potential
        if tag==-1:
            x_index, y_index = np.where(Y<b)
            U[x_index, y_index] = 0
        elif tag==1:
            x_index, y_index = np.where((Y>b) | (X<150))
            U[x_index, y_index] = 0
            
        return U
    
    
    @classmethod
    def RampBoundary(self, X, Y, points, P, σ, tag):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~Calculate the potential
        k,b,xx,yy = get_line(points[0], points[1])
        yt = k*X + b
        dist = abs(k*X-Y+b)/sqrt(k**2+1)
        U = P * np.exp( -(dist) / (2*σ**2) )
        
        #~~~~~~~~~~~~~~~~~~~~~~~~Judge the position whether the points are on the dowm/up directio of the boundary
        sign = np.zeros(U.shape)
        for i in range(sign.shape[0]):
            for j in range(sign.shape[1]):
                if Y[i,j] > X[i,j]*k+b:
                    sign[i,j] = 1
                elif Y[i,j] < X[i,j]*k+b:
                    sign[i,j] = -1
                    
       #~~~~~~~~~~~~~~~~~~~~~~~~Remove non-potential 
        if tag==-1:
            x_index, y_index = np.where(sign==1)
            U[x_index, y_index] = 0
        elif tag==1:
            x_index, y_index = np.where((sign==-1) | (X<=150))
            U[x_index, y_index] = 0
            
        return U
    
    
    @classmethod
    def LaneMarkings(self, X, Y, points, P, σ, tag):
        σ = 1
        #~~~~~~~~~~~~~~~~~~~~~~~~Calculate the potential
        k,b,xx,yy = get_line(points[0], points[1])
        yt = k*X + b
        dist = abs(k*X-Y+b)/sqrt(k**2+1)
        U = P * np.exp( -(dist) / (2*σ**2))
        
        if tag==1:
            U = P * np.exp( -(dist) / (2*σ**2))
            x_index, y_index = np.where((X>=150))
            U[x_index, y_index] = 0
        if tag==3:
            U = 4 * np.exp( -(dist) / (2*σ**2))
            x_index, y_index = np.where(X<=150)
            U[x_index, y_index] = 0
 
        return U
    
    
    @classmethod
    def VirtualBoundary(self, X, Y, xvals, yvals, X0, Y0, σ, α, β):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~Calculate the potential
        U_vbp = np.zeros(X.shape)
        for i in range(X0.shape[0]):
            for j in range(Y0.shape[0]):
                U_vbp[j,i] = Potential_VBP(xvals, yvals, X0[i], Y0[j], α, β)
        
        return U_vbp


## 2.3.**Determines feasible region**
class GetFeasibleRegion:
    '''
        Calculate the feasible region for vehicles based on the topology of the road
    '''
    @classmethod
    def GetWholeRegionRamp(self, X):
        region = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i,j]<= 21.58:                           ### the first part
                    k,b,xx,yy = get_line([0,230],[8.5,16.5])
                    if (Y[i,j]>=0.5) and (Y[i,j] <= k*X[i,j]+b):
                        region[i,j] = 1
                    else:
                        region[i,j] = -1

                elif X[i,j]<=150 and X[i,j] > 21.58:         ### the second part
                    k,b,xx,yy = get_line([0,230],[8.5,16.5])
                    distance = np.sqrt((xvals-X[i,j])**2 + (yvals-Y[i,j])**2)
                    idx = distance.argmin()
                    p_min = xvals[idx], yvals[idx]

                    if (p_min[1]-Y[i,j]<=0) and (Y[i,j] <= k*X[i,j]+b):
                        region[i,j] = 1
                    else:
                        region[i,j] = -1

                else:                                        ### the third part
                    k,b,xx,yy = get_line([0,230],[8.5,16.5])
                    k1,b1,xx,yy = get_line([150,250],[7.6,11])
                    if (Y[i,j] >= k1*X[i,j]+b1) and (Y[i,j] <= k*X[i,j]+b):
                        region[i,j] = 1
                    else:
                        region[i,j] = -1 

        return region
    
    
    @classmethod
    def GetWholeRegionStraight(self, Y):
        region = np.zeros(Y.shape)
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if Y[i,j]<=7.6 and Y[i,j]>=0.5:
                    region[i,j] = 1
                else:
                    region[i,j] = -1
                
        return region
    
    
    @classmethod
    def GetPartRegion(self, region, x_last, y_last, Δx, Δx_min, X, Y):
        sub_region = region
        
        
        ##--------------------------------------------------This is for ramp
        if y_last < 7.6:
            x_index, y_index = np.where((X<=x_last+Δx_min) | (X>=x_last+Δx))        
            sub_region[x_index, y_index] = -1
            
            k,b,xx,yy = get_line( [50, 230], [7.6, 13.5])
            k1,b1,xx,yy = get_line( [150,250],[7.6,11])
            x_index, y_index = np.where((Y <= k*X +b) )     
            sub_region[x_index, y_index] = -1
            
            x_index, y_index = np.where(sub_region==1)
        else:
            x_index, y_index = np.where((X<=x_last) | (X>=x_last+Δx) | (Y<=y_last))    
            x_index, y_index = np.where((X<=x_last+Δx_min) | (X>=x_last+Δx))     
            sub_region[x_index, y_index] = -1
        
            k,b,xx,yy = get_line( [50, 230], [7.6, 13.5])
            k1,b1,xx,yy = get_line( [150,250],[7.6,11])
            x_index, y_index = np.where((Y <= k*X +b) )     
            sub_region[x_index, y_index] = -1
            
            x_index, y_index = np.where(sub_region==1)    
        '''
        ##--------------------------------------------------This is for straight segment
        x_index, y_index = np.where((X<=x_last+Δx_min) | (X>=x_last+Δx) | (Y>3.7))        
        sub_region[x_index, y_index] = -1
        x_index, y_index = np.where(sub_region==1)
        '''    
        return x_index, y_index, sub_region


def get_vehicle_expected_position(x_last, y_last, Δx, Δx_min, X, Y, U_all):
    '''
        Calculate the next expected position for vehicle
    '''
    ## get feasible region
    region = scio.loadmat('region.mat')['data']
    x_index, y_index, sub_region = GetFeasibleRegion.GetPartRegion(region, x_last, y_last, Δx, Δx_min, X, Y)
    
    ## find the expected location
    idx_min = np.argwhere(U_all[x_index, y_index] == np.min(U_all[x_index, y_index]))
    x_e, y_e = X[x_index[idx_min], y_index[idx_min]][0][0], Y[x_index[idx_min], y_index[idx_min]][0][0]
    
    return x_e, y_e