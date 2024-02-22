
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
    if beta > 20 * np.pi / 180:
        beta = 20 * np.pi / 180
    elif beta < -20 * np.pi / 180:
        beta = -20 * np.pi / 180
    return beta   
    

def Get_phi(V, sigma, beta, lr, lf, dt, phi_last):
    '''
        Calculate the 'φ', which is vehicle's heading angle
    '''
    #*********************************Calculate the angle
    phi_new = (V*cos(beta)*tan(sigma) / (lf+lr)) * dt + phi_last
    
    #**********************************Constrain the domain
    if phi_new > 45 *  np.pi / 180:
        phi_new = 45 *np.pi / 180
    elif phi_new < -45 * np.pi / 180:
        phi_new = -45 * np.pi / 180
    
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
    # a_sigma = a_sigma_d * (1 - (v_sigma_last/v_sigma_d)**4 - exp(sigma_last) - Lambda*tan(-alpha))
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
    # alpha = atan(abs(y_e-yt)/(x_e-xt))
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
## 2.2.**Visulization function**

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
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~Plot the arrows
    # ax.arrow(-20, dataframe.y.min()-2, 0, 2, length_includes_head=False,head_width=5, head_length=1.25, fc='r', ec='r') 
    # ax.arrow(-20, dataframe.y.min()-2, 25, 0, length_includes_head=False,head_width=0.75, head_length=10, fc='r', ec='r') 
    # plt.ylim(dataframe.y.min()-5, dataframe.y.max()+5)
    plt.gca().invert_yaxis()

    
def data_columns_selection(dataframe):
    '''
        Select the specific columns
        ---------------------------
    '''
    if dataframe.shape[1]==25:
        dataframe.rename(columns=({'width':'length', 'height':'width'}), inplace=True) 
    else:
        dataframe.rename(columns=({'trackId':'id', 'xCenter':'x', 'yCenter':'y'}), inplace=True) 
    # data = dataframe[['frame','id','x','y','xVelocity','yVelocity','xAcceleration','yAcceleration','laneId','length','width']]
    
    return dataframe


def plot_topology(lane):
    '''
        Plot these lane markings
    '''
    for i in range(len(lane)):
        if i in [1,2,5,6]:
            if i==6:
                plt.axhline(y=lane[i], ls='--',color='k',linewidth=2, label='lane marks')
            else:
                plt.axhline(y=lane[i], ls='--',color='k',linewidth=2)
        else:
            if i==7:
                plt.axhline(y=lane[i], color='k',linewidth=3, label='boundary')
            else:
                plt.axhline(y=lane[i], color='k',linewidth=3)







#######################################################################################################################
#######################################################################################################################
## 2.3.**Data processing function**

def extract_laneBoundary(data):
    '''
        Extract the lane boundary from 
        the full dataset
    '''
    ## get lanes and its conrresponding lane boundary
    lanes = np.sort(data.laneId.unique())

    ## filter lane boundaries
    laneBoundary = {}
    for l in lanes:
        temp = data[data['laneId']==l]
        laneBoundary[l] = [temp.y.describe()['min'], temp.y.describe()['max']]

    ## shift lane boundaries
    for i in range(len(lanes)-1):
        # the local lane
        l0 = lanes[i]
        b0 = laneBoundary[l0]

        # the next lane
        l1 = lanes[i+1]
        b1 = laneBoundary[l1]

        # shift boundaries
        if b0[1]!=b1[0] and b1[0]-b0[1]<=2:
            # take the average value as the mid-boundary
            mid_b = round((b1[0]+b0[1]) / 2, 2)

            # uptate data
            laneBoundary[l0] = [b0[0], mid_b]
            laneBoundary[l1] = [mid_b, b1[1]]
            
    return laneBoundary


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


def extract_scenario_data(Ld, Fd, lanekeep, highd):
    '''
        Extract the scenario data
        -------------------------
    '''
    vids = lanekeep.id.unique()

    RES = []

    for id in tqdm(vids):
        veh = highd[highd['id']==id]
        for i in range(veh.shape[0]):
            ## get current vehicle status
            veh.iloc[i]
            t, x = veh.iloc[i][['frame', 'x']]

            ## front vehicle
            veh_f = highd[(highd.id!=id) & (highd.frame==t) & (highd.laneId==7) & (highd.x>x) & (highd.x<=x+Fd)]

            ## left-front vehicle
            veh_lf = highd[(highd.id!=id) & (highd.frame==t) & (highd.laneId==8) & (highd.x>x) & (highd.x<=x+Ld)]

            ## right-front vehicle
            veh_rf = highd[(highd.id!=id) & (highd.frame==t) & (highd.laneId==6) & (highd.x>x) & (highd.x<=x+Ld)]

            ## filter useful data
            surroundings = veh.iloc[i].T.tolist()
            for sv in [veh_f, veh_lf, veh_rf]:
                if sv.shape[0]!=0:
                    res = sv.iloc[0][['id', 'x', 'y', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']].tolist()
                else:
                    res = [None, None, None, None, None, None, None]
            surroundings = surroundings + res
            
            ## save the data
            RES.append(res)

    ## create columns
    columns = highd.columns.tolist()
    for h in ['f_', 'lf_', 'rf_']:
        for t in ['id', 'x', 'y', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']:
            columns.append(h+t)


    ## create columns
    RES = pd.DataFrame(np.array(RES), columns=columns)
    
    return RES



