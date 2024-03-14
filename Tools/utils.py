import numpy as np
from scipy.interpolate import splrep, splev
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import enum
from typing import List
from datetime import datetime
import json
from skimage.measure import CircleModel,LineModelND, ransac
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor



class SS_VARIABLE(enum.Enum):
    X = 0
    Y = 1
    Z = 2
    Vx = 3
    Vy = 4
    Vz = 5
    Px = 3
    Py = 4
    Pz = 5

class Trajectory_Source(enum.Enum):
    Amit_Simulated = 0
    Yassid_Simulated = 1
    Experiment = 2

class Trajectory_SS_Type(enum.Enum):
    Real = "Real SS"
    Estimated_FW = "Estimated FW SS"
    Estimated_BW = "Estimated BW SS"
    Observed = "Observed"

Q_PROTON = torch.tensor(1.6022*1e-19)
MASS_PROTON_KG = torch.tensor(1.6726*1e-27)
MASS_PROTON_AMU = torch.tensor(1.0072766)
CM_NS__TO__M_S = 1e7
M_S__TO__CM_NS = 1e-7
M_S_SQUARED__TO__CM_NS_SQUARED = 100 * (1e-9)**2
CM__TO__M = 0.01
B = 2.85 #Applied Magnetic Field (T)
E = torch.cos((Q_PROTON*B)/MASS_PROTON_KG) * 500 #Applied Electric Field (V/m)
ATOMIC_NUMBER = 1
C = 3*1e8
CHAMBER_RADIUS  = 25 #cm




class CONFIG():
    def __init__(self,config_path) -> None:
        self.parse_config(config_path)
    def parse_config(self,config_path):
        with open(config_path, 'r') as f:
            data = json.load(f)
        for key,value in data.items():
            setattr(self,key,value)
        return data

class Trajectory():
    def __init__(self,real_traj_data,observation_data=None,data_source:Trajectory_Source = Trajectory_Source.Amit_Simulated,momentum_ss=False,init_energy=None,init_teta=None,init_phi=None,delta_t=None) -> None:
        assert not(momentum_ss), "No support for momentum SS Yet!"
        assert delta_t is not None or len(real_traj_data['t']) > 0, "delta_t or time stamps are needed for propagation function!"

        self.data_src = data_source
        self.init_energy = init_energy
        self.init_teta = init_teta
        self.init_phi = init_phi
        self.t = real_traj_data['t'] if 't' in real_traj_data else torch.tensor([])
        self.delta_t = delta_t if delta_t is not None else self.t[1]-self.t[0]
        self.momentum_ss = momentum_ss
        self.real_energy = real_traj_data['energy'] if 'energy' in real_traj_data else torch.tensor([])
        if momentum_ss:
            self.x_real = torch.cat((real_traj_data['x'],real_traj_data['y'],real_traj_data['z'],real_traj_data['px'],real_traj_data['py'],real_traj_data['pz']),dim=1).T
        else:
            self.x_real = torch.cat((real_traj_data['x'],real_traj_data['y'],real_traj_data['z'],real_traj_data['vx'],real_traj_data['vy'],real_traj_data['vz']),dim=1).T
        if observation_data is not None:
            self.y = torch.cat(observation_data['x'],observation_data['y'],observation_data['z'],dim=1).T
        else:
            #TODO add noise
            noise = 0
            self.y = self.x_real[[SS_VARIABLE.X.value,SS_VARIABLE.Y.value,SS_VARIABLE.Z.value],:] + noise

        self.traj_length = self.x_real.shape[1]

        self.x_estimated_FW = torch.zeros_like(self.x_real)
        self.x_estimated_BW = torch.zeros_like(self.x_real)
        self.energy_estimated_FW = torch.zeros_like(self.x_real)
        self.energy_estimated_BW = torch.zeros_like(self.x_real)
    
    def set_name(self,name):
        self.traj_name = name

    def traj_plots(self,SS_to_plot : List[Trajectory_SS_Type],show=True,save=False,output_path=None):
        space_state_vector_list = []
        energy_list = []
        for traj_ss_type in SS_to_plot:
            if traj_ss_type == Trajectory_SS_Type.Real:
                space_state_vector_list.append(self.x_real)
                energy_list.append(self.real_energy)
            elif traj_ss_type == Trajectory_SS_Type.Estimated_FW:
                space_state_vector_list.append(self.x_estimated_FW)
                energy_list.append(self.energy_estimated_FW)
            elif traj_ss_type == Trajectory_SS_Type.Estimated_BW:
                space_state_vector_list.append(self.x_estimated_BW)
                energy_list.append(self.energy_estimated_BW)
            elif traj_ss_type == Trajectory_SS_Type.Observed:
                space_state_vector_list.append(self.y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i,space_state_vector in enumerate(space_state_vector_list):
            ax.scatter3D(space_state_vector[SS_VARIABLE.X.value,:], space_state_vector[SS_VARIABLE.Y.value,:], space_state_vector[SS_VARIABLE.Z.value,:],label=SS_to_plot[i].value)
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectory')

        fig, axs = plt.subplots(2)  # 2 rows of subplots
        for space_state_vector in space_state_vector_list:
            axs[0].plot(space_state_vector[SS_VARIABLE.Vx.value,:],label=f'x {SS_to_plot[i].value}')
            axs[0].plot(space_state_vector[SS_VARIABLE.Vy.value,:],label=f'y {SS_to_plot[i].value}')
            axs[0].plot(space_state_vector[SS_VARIABLE.Vz.value,:],label=f'z {SS_to_plot[i].value}')
        axs[0].set_title(f"{'Momentums' if self.momentum_ss else 'Velocities'} Over Time")
        axs[0].set_ylabel(f"{'Momentums [GeV/c]' if self.momentum_ss else 'Velocity [m/s]'}")
        axs[0].set_xticks([])
        axs[0].legend()
        for i,energy in enumerate(energy_list):
            axs[1].plot(self.t,energy.flatten(),label=f'Energy {SS_to_plot[i].value}')
        axs[1].set_title('Kinetic Energy Over Time')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Energy [MeV]')
        axs[1].legend()
        if show:
            plt.show()

class Traj_Generator():
    def __init__(self,max_traj_length = 1000) -> None:
        self.max_traj_length = max_traj_length
        self.x = torch.zeros((self.max_traj_length ,1))
        self.y = torch.zeros((self.max_traj_length ,1))
        self.z = torch.zeros((self.max_traj_length ,1))
        self.vx = torch.zeros((self.max_traj_length ,1))
        self.vy = torch.zeros((self.max_traj_length ,1))
        self.vz = torch.zeros((self.max_traj_length ,1))
        self.t = torch.zeros((self.max_traj_length ,1))
        self.energy = torch.zeros((self.max_traj_length ,1))
        self.delta_t = 0.5 #step size in nseconds

    def set_init_values(self,energy=None,theta=None,init_vx=None,init_vy=None,init_vz=None,phi=0,init_x=0,init_y=0,init_z=0):
        self.init_energy = energy
        self.init_teta = theta
        self.init_phi = phi
        self.x[0]  = init_x
        self.y[0]  = init_y
        self.z[0]  = init_z
        if energy is None:
            self.vx[0] = init_vx
            self.vz[0] = init_vy
            self.vy[0] = init_vz
        else:
            M_Ener = MASS_PROTON_AMU * 931.49401
            # E = sqrt(p^2+ M_ener^2) - M_ener
            p = torch.sqrt((energy + M_Ener)**2 - M_Ener**2) # MeV/c
            v = convert_momentum_to_velocity(p)
            self.vx[0] = v * np.sin(theta) * np.cos(phi)
            self.vy[0] = v * np.sin(theta) * np.sin(phi)
            self.vz[0] = v * np.cos(theta)

    def generate(self,energy=None,theta=None,phi = 0,init_x = 0,init_y = 0,init_z = 0,init_vx=None,init_vy=None,init_vz=None):
        self.set_init_values(energy=energy,theta=theta,phi=phi,init_x=init_x,
                             init_y=init_y,init_z=init_z,init_vx=init_vx,
                             init_vy=init_vy,init_vz=init_vz)

        self.energy[0] = curr_energy = get_energy_from_velocities(self.vx[0],self.vy[0],self.vz[0])
        i=1
        while (curr_energy > self.energy[0] * 0.01 and i<self.max_traj_length):

            state_space_vector_prev= (self.x[i-1],self.y[i-1],self.z[i-1],self.vx[i-1],self.vy[i-1],self.vz[i-1])
            state_space_vector_curr = f(state_space_vector_prev,self.delta_t)
            (self.x[i],self.y[i],self.z[i],self.vx[i],self.vy[i],self.vz[i]) = state_space_vector_curr

            self.t[i] = i * self.delta_t
            self.energy[i] = curr_energy = get_energy_from_velocities(self.vx[i],self.vy[i],self.vz[i])
            distance_from_z_axis = torch.sqrt(self.x[i]**2 + self.y[i])
            if distance_from_z_axis >= CHAMBER_RADIUS:
                break
            i+=1

        traj_dict = {
            "t" : self.t[:i-1],
            "x" : self.x[:i-1],
            "y" : self.y[:i-1],
            "z" : self.z[:i-1],
            "vx" : self.vx[:i-1],
            "vy" : self.vy[:i-1],
            "vz" : self.vz[:i-1],
            "px" : convert_velocity_to_momentum(self.vx[:i-1]),
            "py" : convert_velocity_to_momentum(self.vy[:i-1]),
            "pz" : convert_velocity_to_momentum(self.vz[:i-1]),
            "energy" : self.energy[:i-1],
        }
        traj = Trajectory(real_traj_data=traj_dict,init_energy=self.init_energy,init_teta=self.init_teta,init_phi=self.init_phi)
        get_mx_0(traj.x_real)
        return traj

def get_energy_from_brho(brho):
    '''
    Input : 
        brho [Tm]
    Output : 
        energy - [MeV]
        p - [MeV/c]
    '''
    M_Ener = MASS_PROTON_AMU * 931.49401 #MeV
    p = brho * ATOMIC_NUMBER * (2.99792458 * 100) #MeV/c
    energy = np.sqrt(p**2 + M_Ener**2) - M_Ener
    return energy,p

def plot_circle_with_fit(x_center_fit, y_center_fit, radius_fit,traj_x,traj_y):
    theta = np.linspace(0, 2*np.pi, 100)  # Create 100 points around the circumference
    x_fit = x_center_fit + radius_fit * np.cos(theta)  # Calculate x coordinates of points
    y_fit = y_center_fit + radius_fit * np.sin(theta)  # Calculate y coordinates of points
    plt.figure()
    plt.scatter(x_fit,y_fit,label="fit",color='red')
    plt.plot(traj_x,traj_y,label="true traj")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()



def get_mx_0(traj_coordinates):
    mx_0 = torch.zeros(6) #Size of state vector is 6x1
    x = traj_coordinates[SS_VARIABLE.X.value,:]
    y = traj_coordinates[SS_VARIABLE.Y.value,:]
    z = traj_coordinates[SS_VARIABLE.Z.value,:]

    NUM_POINTS = traj_coordinates.shape[1]

    model, inliers = ransac(traj_coordinates[[SS_VARIABLE.X.value,SS_VARIABLE.Y.value],:].numpy().T, CircleModel, min_samples=max(3,int(NUM_POINTS*0.1)), residual_threshold=6, max_trials=1000)
    x_center = model.params[0] 
    y_center = model.params[1] 
    init_radius = model.params[2] * CM__TO__M
    plot_circle_with_fit(x_center * CM__TO__M,y_center * CM__TO__M,init_radius,x * CM__TO__M,y* CM__TO__M)




    y_from_center = y - y_center
    x_from_center = x - x_center

    phis = np.unwrap(torch.arctan2(y_from_center,x_from_center))
    phis = phis[0] - phis
    arc_lengths  = phis * init_radius

    ransacc = RANSACRegressor(LinearRegression(),min_samples=max(2,int(NUM_POINTS*0.1)),residual_threshold=6.0,max_trials=1000)
    ransacc.fit(arc_lengths.reshape(-1,1), z.numpy() * CM__TO__M)
    vector = torch.tensor([1,ransacc.estimator_.intercept_ + ransacc.estimator_.coef_[0]])
    vector /= torch.norm(vector,p=2)
    

    ## Init Angles ##
    init_theta = torch.arccos(vector[1])
    init_phi = torch.arctan2(y[1]-y[0],x[1]-x[0])


    ### Init Energy ###
    brho = init_radius * B / np.sin(init_theta)
    init_energy,init_p = get_energy_from_brho(brho)

    mx_0[SS_VARIABLE.X.value] = x[0]
    mx_0[SS_VARIABLE.Y.value] = y[0]
    mx_0[SS_VARIABLE.Z.value] = z[0]
    mx_0[SS_VARIABLE.Vx.value] = convert_momentum_to_velocity(init_p) * np.sin(init_theta) * np.cos(init_phi)
    mx_0[SS_VARIABLE.Vy.value] = convert_momentum_to_velocity(init_p) * np.sin(init_theta) * np.sin(init_phi)
    mx_0[SS_VARIABLE.Vz.value] = convert_momentum_to_velocity(init_p) * np.cos(init_theta)
    return mx_0



def convert_momentum_to_velocity(p):
    '''
    Input : 
        p [MeV]
    Output : 
        v [cm/ns]
    '''
    v = (p * 5.344286e-22 / MASS_PROTON_KG) *  M_S__TO__CM_NS#cm/ns
    return v

def convert_velocity_to_momentum(v):
    '''
    Input : 
        v [cm/ns]
    Output : 
        p [MeV]
    '''
    p = (v * CM_NS__TO__M_S * MASS_PROTON_KG)/ 5.344286e-22 #MeV/c
    return p

def get_energy_from_velocities(vx,vy,vz):
    '''
    Input : 
        vx - [cm/ns]
        vy - [cm/ns]
        vz - [cm/ns]
    Output :
        energy - [MeV]
    '''
    velocity = torch.sqrt(vx**2 + vy**2 + vz**2) * CM_NS__TO__M_S
    bet = velocity / C
    gamma = torch.sqrt(1/(1-bet**2)) 
    energy = (gamma - 1) * 931.494028
    return energy 

def get_vel_deriv(vx,vy,vz,direction):
    '''
    Input : 
        vx - [cm/ns]
        vy - [cm/ns]
        vz - [cm/ns]
        direction - 'x' / 'y' / 'z'
    Output :
        a - [cm/ns^2]
    '''
    energy = get_energy_from_velocities(vx,vy,vz)
    #convert velocities to m/s for computation
    temp_vx = vx * CM_NS__TO__M_S
    temp_vy = vy * CM_NS__TO__M_S
    temp_vz = vz * CM_NS__TO__M_S
    deaccel = get_deacceleration(energy)
    Bx = 0
    By = 0
    Bz = B
    Ex = 0
    Ey = 0
    Ez = -E

    rr = torch.sqrt(vx**2 + vy**2 + vz**2)
    az = torch.arctan2(vy,vx)
    po = torch.arccos(vz/rr)
    if direction == 'x':
        a = (Q_PROTON/MASS_PROTON_KG) * (Ex + temp_vy*Bz-temp_vz*By) - deaccel*torch.sin(po)*torch.cos(az)
    elif direction == 'y':
        a = (Q_PROTON/MASS_PROTON_KG) * (Ey + temp_vz*Bx - temp_vx*Bz) - deaccel*torch.sin(po)*torch.sin(az)
    elif direction == 'z':
        a = (Q_PROTON/MASS_PROTON_KG) * (Ez + temp_vx*By - temp_vy*Bx) - deaccel*torch.cos(po)
    a *= M_S_SQUARED__TO__CM_NS_SQUARED
    return a

def f(state_space_vector_prev,delta_t):
    x,y,z,vx,vy,vz = state_space_vector_prev

    ## f1 ##
    k1x = vx
    k1y = vy
    k1z = vz

    k1vx = get_vel_deriv(vx,vy,vz,direction='x')
    k1vy = get_vel_deriv(vx,vy,vz,direction='y')
    k1vz = get_vel_deriv(vx,vy,vz,direction='z')

    ## f2 ##
    k2x = vx + 0.5 * delta_t * k1x
    k2y = vy + 0.5 * delta_t * k1y
    k2z = vz + 0.5 * delta_t * k1z

    k2vx = get_vel_deriv(vx + 0.5*delta_t*k1vx,vy + 0.5*delta_t*k1vy,vz+ 0.5*delta_t*k1vz,direction='x')
    k2vy = get_vel_deriv(vx + 0.5*delta_t*k1vx,vy + 0.5*delta_t*k1vy,vz+ 0.5*delta_t*k1vz,direction='y')
    k2vz = get_vel_deriv(vx + 0.5*delta_t*k1vx,vy + 0.5*delta_t*k1vy,vz+ 0.5*delta_t*k1vz,direction='z')

    ## f3 ##
    k3x = vx + 0.5 * delta_t * k2x
    k3y = vy + 0.5 * delta_t * k2y
    k3z = vz + 0.5 * delta_t * k2z

    k3vx = get_vel_deriv(vx + 0.5*delta_t*k2vx,vy + 0.5*delta_t*k2vy,vz+ 0.5*delta_t*k2vz,direction='x')
    k3vy = get_vel_deriv(vx + 0.5*delta_t*k2vx,vy + 0.5*delta_t*k2vy,vz+ 0.5*delta_t*k2vz,direction='y')
    k3vz = get_vel_deriv(vx + 0.5*delta_t*k2vx,vy + 0.5*delta_t*k2vy,vz+ 0.5*delta_t*k2vz,direction='z')

    ## f4 ##
    k4x = vx + delta_t * k3x
    k4y = vy + delta_t * k3y
    k4z = vz + delta_t * k3z

    k4vx = get_vel_deriv(vx + delta_t*k3vx,vy + delta_t*k3vy,vz+ delta_t*k3vz,direction='x')
    k4vy = get_vel_deriv(vx + delta_t*k3vx,vy + delta_t*k3vy,vz+ delta_t*k3vz,direction='y')
    k4vz = get_vel_deriv(vx + delta_t*k3vx,vy + delta_t*k3vy,vz+ delta_t*k3vz,direction='z')

    vx = vx + (delta_t/6) * (k1vx+ 2*k2vx + 2*k3vx + k4vx)
    vy = vy + (delta_t/6) * (k1vy + 2*k2vy + 2*k3vy + k4vy)
    vz = vz + (delta_t/6) * (k1vz + 2*k2vz + 2*k3vz + k4vz)

    x = x + (delta_t/6) * (k1x + 2*k2x + 2*k3x + k4x)
    y = y + (delta_t/6) * (k1y + 2*k2y + 2*k3y + k4y)
    z = z + (delta_t/6) * (k1z + 2*k2z + 2*k3z + k4z)

    state_space_vector_curr = (x,y,z,vx,vy,vz)
    return state_space_vector_curr

def h(space_state_vector):
    ''' 
    INPUT:
        space_state_vector - shape of [batch_size,space_state_vector_size,1]
    OUTPUT:
        space_state_vector - shape of [batch_size,observation_vector_size,1]

    '''
    H = torch.zeros(3,space_state_vector.shape[1])
    H[0,0] = H[1,1] = H[2,2] = 1
    obs_vector = torch.matmul(H,space_state_vector)
    return obs_vector

def get_deacceleration(energy_interp):
    '''
    Input : 
        energy_interp - [MeV]
    Output :
        interp_stopping_acc - [m/s^2]
    '''
    gasMediumDensity = 8.988e-5 #g/cm3 at 1 bar
    data = np.loadtxt("/Users/amitmilstein/Documents/Ben_Gurion_Univ/MSc/TPC_RTSNet/TPC_Reconstruction/Tools/stpHydrogen.txt")
    energy = data[:, 0]  # First column
    stopping_power = data[:, 1]  # Second column
    tck = splrep(energy, stopping_power)
    interp_stopping_power = splev(energy_interp, tck)

    interp_stopping_force =interp_stopping_power * 1.6021773349e-13 * gasMediumDensity*100
    interp_stopping_acc = interp_stopping_force / MASS_PROTON_KG
    return interp_stopping_acc

def generate_dataset(N_Train,N_Test,N_CV,dataset_name = "dataset",output_dir = "TPC_Reconstruction/Simulations/Particle_Tracking/data"):
    assert (N_Train+N_Test+N_CV)<=2000, "Maximum of 2k Trajectories can be made"
    time_stamp = datetime.now().strftime("_%d_%m_%y__%H_%M")
    dataset_name = dataset_name + time_stamp
    os.makedirs(output_dir,exist_ok=True)
    assert dataset_name not in os.listdir(output_dir), f"Dataset with name '{dataset_name}' Exists!"

    ## Get Angle-Energy
    data = np.loadtxt("TPC_Reconstruction/Tools/Angle_Energy.txt")
    theta = np.radians(data[:, 0]) 
    energy = data[:, 1]  #MeV
    np.random.shuffle(permutation :=np.arange(1, len(energy)))
    theta = theta[permutation].reshape(-1)
    energy = energy[permutation].reshape(-1)
    phi = np.random.uniform(-np.pi, np.pi,len(theta))

    generator = Traj_Generator()
    Dataset = []
    for i in range(N_Train + N_CV + N_Test):
        print(f"Generating trajectory {i}; Energy - {energy[i]},Theta - {theta[i]},Phi - {phi[i]}")
        Dataset.append(generator.generate(energy=energy[i],theta=theta[i],phi=phi[i]))


    training_set = Dataset[:N_Train]
    Dataset = Dataset[N_Train:]
    CV_set =  Dataset[:N_CV]
    test_set = Dataset[N_CV:]
    torch.save([training_set,CV_set,test_set], os.path.join(output_dir,f'{dataset_name}.pt'))
    

if __name__ == "__main__":
    # generate_dataset(N_Train=150,N_CV=25,N_Test=25)
    gen = Traj_Generator()
    traj = gen.generate(energy=30,theta=1,phi=0)
    traj.traj_plots([Trajectory_SS_Type.Real])
    df = pd.DataFrame(traj.x_real.numpy().T,columns = ['x','y','z','vx','vy','vz'])
    df.to_csv('debug_traj_energy_30_teta_phi_0_new.csv', index=False)
        

    # gen.save_csv(traj_data)
