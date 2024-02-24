import numpy as np
import numpy as np
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt

Q_ELECTRON = 1.6022*10e-19
M_ELECTRON = 1.6726*10e-27
B = -3 #Applied Magnetic Field (T)
E = np.cos((Q_ELECTRON*B)/M_ELECTRON) * 500 #Applied Electric Field (V/m)


class Traj_Generator():
    def __init__(self,traj_length = 1000) -> None:
        self.traj_length = traj_length
        self.x = np.zeros((self.traj_length ,1))
        self.y = np.zeros((self.traj_length ,1))
        self.z = np.zeros((self.traj_length ,1))
        self.vx = np.zeros((self.traj_length ,1))
        self.vy = np.zeros((self.traj_length ,1))
        self.vz = np.zeros((self.traj_length ,1))
        self.t = np.zeros((self.traj_length ,1))
        self.energy = np.zeros((self.traj_length ,1))
        self.delta_t = 10e-10 #step size in seconds
        self.set_init_values()

    def set_init_values(self):
        self.vx[0] = 10e6
        self.vz[0] = 10e6
        self.vy[0] = 10e6

    def get_energy(self,vx,vy,vz):
        c = 3*10e8
        velocity = np.sqrt(vx**2 + vy**2 + vz**2)
        bet = velocity / c
        gamma = np.sqrt(1/(1-bet**2))
        energy = (gamma - 1) * 931.494028
        return energy

    def get_stopping_power(self,energy_interp):
        gasMediumDensity = 8.988e-5 #g/cm3 at 1 bar
        data = np.loadtxt("TPC_Reconstruction/Tools/stpHydrogen.txt")
        energy = data[:, 0]  # First column
        stopping_power = data[:, 1]  # Second column
        tck = splrep(energy, stopping_power)
        interp_stopping_power = splev(energy_interp, tck)

        interp_stopping_power *=1.6021773349e-13
        interp_stopping_power *= gasMediumDensity*100
        interp_stopping_power /= M_ELECTRON
        return interp_stopping_power

    def get_vel_deriv(self,vx,vy,vz,direction):
        energy = self.get_energy(vx,vy,vz)
        st = self.get_stopping_power(energy)
        Bx = 0
        By = 0
        Bz = B
        Ex = 0
        Ey = 0
        Ez = -E

        rr = np.sqrt(vx**2 + vy**2 + vz**2)
        az = np.arctan2(vy,vx)
        po = np.arccos(vz/rr)
        if direction == 'x':
            f = (Q_ELECTRON/M_ELECTRON) * (Ex + vy*Bz-vz*By) - st*np.sin(po)*np.cos(az)
        elif direction == 'y':
            f = (Q_ELECTRON/M_ELECTRON) * (Ey + vz*Bx - vx*Bz) - st*np.sin(po)*np.sin(az)
        elif direction == 'z':
            f = (Q_ELECTRON/M_ELECTRON) * (Ez + vx*By - vy*Bx) - st*np.cos(po)
        return f


    def generate(self):
        self.energy[0] = self.get_energy(self.vx[0],self.vy[0],self.vz[0])
        i=0
        while (i < self.traj_length-1):
            k1x = self.vx[i]
            k1y = self.vy[i]
            k1z = self.vz[i]

            k1vx = self.get_vel_deriv(self.vx[i],self.vy[i],self.vz[i],direction='x')
            k1vy = self.get_vel_deriv(self.vx[i],self.vy[i],self.vz[i],direction='y')
            k1vz = self.get_vel_deriv(self.vx[i],self.vy[i],self.vz[i],direction='z')

            k2x = self.vx[i] + 0.5 * self.delta_t * k1x
            k2y = self.vy[i] + 0.5 * self.delta_t * k1y
            k2z = self.vz[i] + 0.5 * self.delta_t * k1z

            k2vx = self.get_vel_deriv(self.vx[i] + 0.5*self.delta_t*k1vx,self.vy[i] + 0.5*self.delta_t*k1vy,self.vz[i]+ 0.5*self.delta_t*k1vz,direction='x')
            k2vy = self.get_vel_deriv(self.vx[i] + 0.5*self.delta_t*k1vx,self.vy[i] + 0.5*self.delta_t*k1vy,self.vz[i]+ 0.5*self.delta_t*k1vz,direction='y')
            k2vz = self.get_vel_deriv(self.vx[i] + 0.5*self.delta_t*k1vx,self.vy[i] + 0.5*self.delta_t*k1vy,self.vz[i]+ 0.5*self.delta_t*k1vz,direction='z')

            k3x = self.vx[i] + 0.5 * self.delta_t * k2x
            k3y = self.vy[i] + 0.5 * self.delta_t * k2y
            k3z = self.vz[i] + 0.5 * self.delta_t * k2z

            k3vx = self.get_vel_deriv(self.vx[i] + 0.5*self.delta_t*k2vx,self.vy[i] + 0.5*self.delta_t*k2vy,self.vz[i]+ 0.5*self.delta_t*k2vz,direction='x')
            k3vy = self.get_vel_deriv(self.vx[i] + 0.5*self.delta_t*k2vx,self.vy[i] + 0.5*self.delta_t*k2vy,self.vz[i]+ 0.5*self.delta_t*k2vz,direction='y')
            k3vz = self.get_vel_deriv(self.vx[i] + 0.5*self.delta_t*k2vx,self.vy[i] + 0.5*self.delta_t*k2vy,self.vz[i]+ 0.5*self.delta_t*k2vz,direction='z')

            k4x = self.vx[i] + self.delta_t * k3x
            k4y = self.vy[i] + self.delta_t * k3y
            k4z = self.vz[i] + self.delta_t * k3z

            k4vx = self.get_vel_deriv(self.vx[i] + self.delta_t*k3vx,self.vy[i] + self.delta_t*k3vy,self.vz[i]+ self.delta_t*k3vz,direction='x')
            k4vy = self.get_vel_deriv(self.vx[i] + self.delta_t*k3vx,self.vy[i] + self.delta_t*k3vy,self.vz[i]+ self.delta_t*k3vz,direction='y')
            k4vz = self.get_vel_deriv(self.vx[i] + self.delta_t*k3vx,self.vy[i] + self.delta_t*k3vy,self.vz[i]+ self.delta_t*k3vz,direction='z')

            #Set next time stamp init value to that of this time step
            self.vx[i] = self.vx[i+1] = self.vx[i] + (self.delta_t/6) * (k1vx+ 2*k2vx + 2*k3vx + k4vx)
            self.vy[i] = self.vy[i+1] = self.vy[i] + (self.delta_t/6) * (k1vy + 2*k2vy + 2*k3vy + k4vy)
            self.vz[i] = self.vz[i+1] = self.vz[i] + (self.delta_t/6) * (k1vz + 2*k2vz + 2*k3vz + k4vz)

            #Set next time stamp init value to that of this time step
            self.x[i] = self.x[i+1] = self.x[i] + (self.delta_t/6) * (k1x + 2*k2x + 2*k3x + k4x)
            self.y[i] = self.y[i+1]  = self.y[i] + (self.delta_t/6) * (k1y + 2*k2y + 2*k3y + k4y)
            self.z[i] = self.z[i+1]  = self.z[i] + (self.delta_t/6) * (k1z + 2*k2z + 2*k3z + k4z)

            self.t[i] = i * self.delta_t

            self.energy[i] = self.get_energy(self.vx[i],self.vy[i],self.vz[i])
            i+=1
        self.energy[i] = self.get_energy(self.vx[i],self.vy[i],self.vz[i])

        state_vector_matrix = np.vstack((self.x.T, self.y.T, self.z.T, self.vx.T, self.vy.T,self.vz.T))
        return state_vector_matrix



gen = Traj_Generator()
a = gen.generate()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(a[0,:], a[1,:], a[2,:], 'gray')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Trajectory')
plt.figure()
plt.plot(a[3,:],label='x velocity')
plt.plot(a[4,:],label='y velocity')
plt.plot(a[5,:],label='z velocity')
plt.legend()
plt.figure()
plt.plot(gen.energy,label='Energy')
plt.legend()
plt.show()
b=1
