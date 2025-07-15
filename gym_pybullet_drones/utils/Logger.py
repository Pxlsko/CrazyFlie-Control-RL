import os
from datetime import datetime
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Logger(object):
    """A class for logging and visualization.

    Stores, saves to file, and plots the kinematic information and RPMs
    of a simulation with one or more drones.

    """

    ################################################################################

    def __init__(self,
                 logging_freq_hz: int,
                 output_folder: str="results",
                 num_drones: int=1,
                 duration_sec: int=0,
                 colab: bool=False,
                 ):
        """Logger class __init__ method.

        Note: the order in which information is stored by Logger.log() is not the same
        as the one in, e.g., the obs["id"]["state"], check the implementation below.

        Parameters
        ----------
        logging_freq_hz : int
            Logging frequency in Hz.
        num_drones : int, optional
            Number of drones.
        duration_sec : int, optional
            Used to preallocate the log arrays (improves performance).

        """
        self.COLAB = colab
        self.OUTPUT_FOLDER = output_folder
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.mkdir(self.OUTPUT_FOLDER)
        self.LOGGING_FREQ_HZ = logging_freq_hz
        self.NUM_DRONES = num_drones
        self.PREALLOCATED_ARRAYS = False if duration_sec == 0 else True
        self.counters = np.zeros(num_drones)
        self.timestamps = np.zeros((num_drones, duration_sec*self.LOGGING_FREQ_HZ))
        #### Note: this is the suggest information to log ##############################
        self.states = np.zeros((num_drones, 16, duration_sec*self.LOGGING_FREQ_HZ)) #### 16 states: pos_x,
                                                                                                  # pos_y,
                                                                                                  # pos_z,
                                                                                                  # vel_x,
                                                                                                  # vel_y,
                                                                                                  # vel_z,
                                                                                                  # roll,
                                                                                                  # pitch,
                                                                                                  # yaw,
                                                                                                  # ang_vel_x,
                                                                                                  # ang_vel_y,
                                                                                                  # ang_vel_z,
                                                                                                  # rpm0,
                                                                                                  # rpm1,
                                                                                                  # rpm2,
                                                                                                  # rpm3
        #### Note: this is the suggest information to log ##############################
        self.controls = np.zeros((num_drones, 12, duration_sec*self.LOGGING_FREQ_HZ)) #### 12 control targets: pos_x,
                                                                                                             # pos_y,
                                                                                                             # pos_z,
                                                                                                             # vel_x, 
                                                                                                             # vel_y,
                                                                                                             # vel_z,
                                                                                                             # roll,
                                                                                                             # pitch,
                                                                                                             # yaw,
                                                                                                             # ang_vel_x,
                                                                                                             # ang_vel_y,
                                                                                                             # ang_vel_z

        self.targets = np.zeros((num_drones, 3, duration_sec*self.LOGGING_FREQ_HZ)) # NUEVO: Array para almacenar las posiciones objetivo


    ################################################################################

    def log(self,
            drone: int,
            timestamp,
            state,
            control=np.zeros(12),
            target_pos=np.zeros(3)
            ):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        drone : int
            Id of the drone associated to the log entry.
        timestamp : float
            Timestamp of the log in simulation clock.
        state : ndarray
            (20,)-shaped array of floats containing the drone's state.
        control : ndarray, optional
            (12,)-shaped array of floats containing the drone's control target.

        """
        if drone < 0 or drone >= self.NUM_DRONES or timestamp < 0 or len(state) != 20 or len(control) != 12:
            print("[ERROR] in Logger.log(), invalid data")
        current_counter = int(self.counters[drone])
        #### Add rows to the matrices if a counter exceeds their size
        if current_counter >= self.timestamps.shape[1]:
            self.timestamps = np.concatenate((self.timestamps, np.zeros((self.NUM_DRONES, 1))), axis=1)
            self.states = np.concatenate((self.states, np.zeros((self.NUM_DRONES, 16, 1))), axis=2)
            self.controls = np.concatenate((self.controls, np.zeros((self.NUM_DRONES, 12, 1))), axis=2)
            self.targets = np.concatenate((self.targets, np.zeros((self.NUM_DRONES, 3, 1))), axis=2) # NUEVO: Extender array de targets

        #### Advance a counter is the matrices have overgrown it ###
        elif not self.PREALLOCATED_ARRAYS and self.timestamps.shape[1] > current_counter:
            current_counter = self.timestamps.shape[1]-1
        #### Log the information and increase the counter ##########
        self.timestamps[drone, current_counter] = timestamp
        #### Re-order the kinematic obs (of most Aviaries) #########
        self.states[drone, :, current_counter] = np.hstack([state[0:3], state[10:13], state[7:10], state[13:20]])
        self.controls[drone, :, current_counter] = control
        self.targets[drone, :, current_counter] = target_pos # NUEVO: Registrar la Position objetivo
        self.counters[drone] = current_counter + 1

    ################################################################################

    def save(self):
        """Save the logs to file.
        """
        with open(os.path.join(self.OUTPUT_FOLDER, "save-flight-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".npy"), 'wb') as out_file:
            np.savez(out_file, timestamps=self.timestamps, states=self.states, controls=self.controls)

    ################################################################################

    def save_as_csv(self,
                    comment: str=""
                    ):
        """Save the logs---on your Desktop---as comma separated values.

        Parameters
        ----------
        comment : str, optional
            Added to the foldername.

        """
        csv_dir = os.path.join(self.OUTPUT_FOLDER, "save-flight-"+comment+"-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir+'/')
        t = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)
        for i in range(self.NUM_DRONES):
            with open(csv_dir+"/x"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 0, :]])), delimiter=",")
            with open(csv_dir+"/y"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 1, :]])), delimiter=",")
            with open(csv_dir+"/z"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 2, :]])), delimiter=",")
            ####
            with open(csv_dir+"/r"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 6, :]])), delimiter=",")
            with open(csv_dir+"/p"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 7, :]])), delimiter=",")
            with open(csv_dir+"/ya"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 8, :]])), delimiter=",")
            ####
            with open(csv_dir+"/rr"+str(i)+".csv", 'wb') as out_file:
                rdot = np.hstack([0, (self.states[i, 6, 1:] - self.states[i, 6, 0:-1]) * self.LOGGING_FREQ_HZ ])
                np.savetxt(out_file, np.transpose(np.vstack([t, rdot])), delimiter=",")
            with open(csv_dir+"/pr"+str(i)+".csv", 'wb') as out_file:
                pdot = np.hstack([0, (self.states[i, 7, 1:] - self.states[i, 7, 0:-1]) * self.LOGGING_FREQ_HZ ])
                np.savetxt(out_file, np.transpose(np.vstack([t, pdot])), delimiter=",")
            with open(csv_dir+"/yar"+str(i)+".csv", 'wb') as out_file:
                ydot = np.hstack([0, (self.states[i, 8, 1:] - self.states[i, 8, 0:-1]) * self.LOGGING_FREQ_HZ ])
                np.savetxt(out_file, np.transpose(np.vstack([t, ydot])), delimiter=",")
            ###
            with open(csv_dir+"/vx"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 3, :]])), delimiter=",")
            with open(csv_dir+"/vy"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 4, :]])), delimiter=",")
            with open(csv_dir+"/vz"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 5, :]])), delimiter=",")
            ####
            with open(csv_dir+"/wx"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 9, :]])), delimiter=",")
            with open(csv_dir+"/wy"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 10, :]])), delimiter=",")
            with open(csv_dir+"/wz"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 11, :]])), delimiter=",")
            ####
            with open(csv_dir+"/rpm0-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 12, :]])), delimiter=",")
            with open(csv_dir+"/rpm1-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 13, :]])), delimiter=",")
            with open(csv_dir+"/rpm2-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 14, :]])), delimiter=",")
            with open(csv_dir+"/rpm3-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 15, :]])), delimiter=",")
            ####
            with open(csv_dir+"/pwm0-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 12, :] - 4070.3) / 0.2685])), delimiter=",")
            with open(csv_dir+"/pwm1-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 13, :] - 4070.3) / 0.2685])), delimiter=",")
            with open(csv_dir+"/pwm2-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 14, :] - 4070.3) / 0.2685])), delimiter=",")
            with open(csv_dir+"/pwm3-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 15, :] - 4070.3) / 0.2685])), delimiter=",")

    
  
    
    def plot(self, pwm=False, reference=None):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        pwm : bool, optional
            If True, converts logged RPM into PWM values (for Crazyflies).

        """
        #### Loop over colors and line styles ######################
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) + cycler('linestyle', ['-', '--', ':', '-.'])))
        fig, axs = plt.subplots(10, 2)
        t = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)

        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        ref_x, ref_y, ref_z = None, None, None
        if reference is not None:
            ref = np.array(reference)
            ref_steps = len(t)
            ref_idx = np.linspace(0, ref_steps-1, len(ref))           
            ref_x = np.interp(np.arange(ref_steps), ref_idx, ref[:,0])
            ref_y = np.interp(np.arange(ref_steps), ref_idx, ref[:,1])
            ref_z = np.interp(np.arange(ref_steps), ref_idx, ref[:,2])

        # XYZ
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 0, :], label="drone_"+str(j))
        if ref_x is not None:
            axs[row, col].plot(t, ref_x, 'b--', label="ref_x")
        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 1, :], label="drone_"+str(j))
        if ref_y is not None:
            axs[row, col].plot(t, ref_y, 'b--', label="ref_y")
        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 2, :], label="drone_"+str(j))
        if ref_z is not None:
            axs[row, col].plot(t, ref_z, 'b--', label="ref_z")
        #### RPY ###################################################
        row = 3
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 6, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r (rad)')
        row = 4
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 7, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('p (rad)')
        row = 5
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 8, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (rad)')

        #### Ang Vel ###############################################
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 9, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wx')
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 10, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wy')
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 11, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wz')

        #### Time ##################################################
        row = 9
        axs[row, col].plot(t, t, label="time")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('time')

        #### Column ################################################
        col = 1

        #### Velocity ##############################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 3, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vx (m/s)')
        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 4, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vy (m/s)')
        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 5, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vz (m/s)')

        #### RPY Rates #############################################
        row = 3
        for j in range(self.NUM_DRONES):
            rdot = np.hstack([0, (self.states[j, 6, 1:] - self.states[j, 6, 0:-1]) * self.LOGGING_FREQ_HZ ])
            axs[row, col].plot(t, rdot, label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('rdot (rad/s)')
        row = 4
        for j in range(self.NUM_DRONES):
            pdot = np.hstack([0, (self.states[j, 7, 1:] - self.states[j, 7, 0:-1]) * self.LOGGING_FREQ_HZ ])
            axs[row, col].plot(t, pdot, label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('pdot (rad/s)')
        row = 5
        for j in range(self.NUM_DRONES):
            ydot = np.hstack([0, (self.states[j, 8, 1:] - self.states[j, 8, 0:-1]) * self.LOGGING_FREQ_HZ ])
            axs[row, col].plot(t, ydot, label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('ydot (rad/s)')

        ### This IF converts RPM into PWM for all drones ###########
        #### except drone_0 (only used in examples/compare.py) #####
        for j in range(self.NUM_DRONES):
            for i in range(12,16):
                if pwm and j > 0:
                    self.states[j, i, :] = (self.states[j, i, :] - 4070.3) / 0.2685

        #### RPMs ##################################################
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 12, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM0')
        else:
            axs[row, col].set_ylabel('RPM0')
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 13, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM1')
        else:
            axs[row, col].set_ylabel('RPM1')
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 14, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM2')
        else:
            axs[row, col].set_ylabel('RPM2')
        row = 9
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 15, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM3')
        else:
            axs[row, col].set_ylabel('RPM3')

        #### Drawing options #######################################
        for i in range (10):
            for j in range (2):
                axs[i, j].grid(True)
                axs[i, j].legend(loc='upper right',
                         frameon=True
                         )
        fig.subplots_adjust(left=0.06,
                            bottom=0.05,
                            right=0.99,
                            top=0.98,
                            wspace=0.15,
                            hspace=0.0
                            )
        if self.COLAB: 
            plt.savefig(os.path.join('results', 'output_figure.png'))
        else:
            plt.show()

            plt.figure(figsize=(10, 6))

        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        rpm_labels = ['RPM0', 'RPM1', 'RPM2', 'RPM3']
        for i in range(4):
            axs[i].plot(t, self.states[0, 12+i, :], label=rpm_labels[i])
            axs[i].set_ylabel(rpm_labels[i])
            axs[i].legend()
            axs[i].grid(True)
        axs[-1].set_xlabel('time (s)')
        plt.suptitle('RPMs de los 4 motores (drone 0)')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

  

        
    def plot_custom(self, reference=None):
        t = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)
        drone = 0

        save_path = r"path/to/save/plots" # Path to save the plots
        os.makedirs(save_path, exist_ok=True)

        # 1. Position
        fig1, axs1 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        
        # X position
        axs1[0].plot(t, self.states[drone, 0, :], 'r-', label="x measured")
        axs1[0].plot(t, self.targets[drone, 0, :], 'b--', label="x desired", drawstyle='steps-post')
        axs1[0].set_ylabel('x (m)')
        axs1[0].legend()
        axs1[0].grid(True)

        # Y position
        axs1[1].plot(t, self.states[drone, 1, :], 'r-', label="y measured")
        axs1[1].plot(t, self.targets[drone, 1, :], 'b--', label="y desired", drawstyle='steps-post')
        axs1[1].set_ylabel('y (m)')
        axs1[1].legend()
        axs1[1].grid(True)

        # Z position
        axs1[2].plot(t, self.states[drone, 2, :], 'r-', label="z measured")
        axs1[2].plot(t, self.targets[drone, 2, :], 'b--', label="z desired", drawstyle='steps-post')
        axs1[2].set_ylabel('z (m)')
        axs1[2].set_xlabel('time (s)')
        axs1[2].legend()
        axs1[2].grid(True)
        
        fig1.suptitle('Desired vs Measured Position')
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        fig1.savefig(os.path.join(save_path, "desired_vs_measured_position.png"))


        # 2. Orientation
        fig2, axs2 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs2[0].plot(t, self.states[drone, 6, :], 'r-', label="roll measured")
        axs2[0].axhline(0, color='b', linestyle='--', label="roll desired") 
        axs2[0].set_ylabel('roll (rad)')
        axs2[0].legend()
        axs2[0].grid(True)
        axs2[1].plot(t, self.states[drone, 7, :], 'r-', label="pitch measured")
        axs2[1].axhline(0, color='b', linestyle='--', label="pitch desired")
        axs2[1].set_ylabel('pitch (rad)')
        axs2[1].legend()
        axs2[1].grid(True)
        axs2[2].plot(t, self.states[drone, 8, :], 'r-', label="yaw measured")
        axs2[2].axhline(0, color='b', linestyle='--', label="yaw desired") 
        axs2[2].set_ylabel('yaw (rad)')
        axs2[2].set_xlabel('time (s)')
        axs2[2].legend()
        axs2[2].grid(True)
        fig2.suptitle('Desired vs Measured Orientation')
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        fig2.savefig(os.path.join(save_path, "desired_vs_measured_orientation.png"))

        # 3. Linear velocity
        fig3, axs3 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs3[0].plot(t, self.states[drone, 3, :], 'r-', label="vx measured")
        axs3[0].axhline(0, color='b', linestyle='--', label="vx desired") 
        axs3[0].set_ylabel('vx (m/s)')
        axs3[0].legend()
        axs3[0].grid(True)
        axs3[1].plot(t, self.states[drone, 4, :], 'r-', label="vy measured")
        axs3[1].axhline(0, color='b', linestyle='--', label="vy desired")
        axs3[1].set_ylabel('vy (m/s)')
        axs3[1].legend()
        axs3[1].grid(True)
        axs3[2].plot(t, self.states[drone, 5, :], 'r-', label="vz measured")
        axs3[2].axhline(0, color='b', linestyle='--', label="vz desired")
        axs3[2].set_ylabel('vz (m/s)')
        axs3[2].set_xlabel('time (s)')
        axs3[2].legend()
        axs3[2].grid(True)
        fig3.suptitle('Desired vs Measured Linear Velocity')
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        fig3.savefig(os.path.join(save_path, "desired_vs_measured_linear_velocity.png"))

        # 4. Angular velocity 
        fig4, axs4 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs4[0].plot(t, self.states[drone, 9, :], 'r-', label="wx measured")
        axs4[0].axhline(0, color='b', linestyle='--', label="wx desired")
        axs4[0].set_ylabel('wx (rad/s)')
        axs4[0].legend()
        axs4[0].grid(True)
        axs4[1].plot(t, self.states[drone, 10, :], 'r-', label="wy measured")
        axs4[1].axhline(0, color='b', linestyle='--', label="wy desired") 
        axs4[1].set_ylabel('wy (rad/s)')
        axs4[1].legend()
        axs4[1].grid(True)
        axs4[2].plot(t, self.states[drone, 11, :], 'r-', label="wz measured")
        axs4[2].axhline(0, color='b', linestyle='--', label="wz desired")
        axs4[2].set_ylabel('wz (rad/s)')
        axs4[2].set_xlabel('time (s)')
        axs4[2].legend()
        axs4[2].grid(True)
        fig4.suptitle('Desired vs Measured Angular Velocity')
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        fig4.savefig(os.path.join(save_path, "desired_vs_measured_angular_velocity.png"))


        # # 5. RPMs without filtering
        # fig5, axs5 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        # rpm_labels = ['RPM0', 'RPM1', 'RPM2', 'RPM3']
        # for i in range(4):
        #     axs5[i].plot(t, self.states[drone, 12+i, :], label=rpm_labels[i], color='red')
        #     axs5[i].set_ylabel(rpm_labels[i])
        #     axs5[i].legend()
        #     axs5[i].grid(True)
        # axs5[-1].set_xlabel('time (s)')
        # fig5.suptitle('Motor\'s RPM')
        # plt.tight_layout(rect=[0, 0.05, 1, 0.96])

        # Butterworth low-pass filter function
        def butter_lowpass_filter(data, cutoff, fs, order=4):
                nyq = 0.5 * fs  # Frecuencia de Nyquist
                normal_cutoff = cutoff / nyq
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                y = filtfilt(b, a, data)
                return y
        
        # 5. RPMs filtered
        fig5, axs5 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        rpm_labels = ['RPM0', 'RPM1', 'RPM2', 'RPM3']

        # Filter parameters
        cutoff = 5.0  # Hz
        fs = self.LOGGING_FREQ_HZ  # Sample frequency

        for i in range(4):
            raw_rpm = self.states[drone, 12+i, :]
            filtered_rpm = butter_lowpass_filter(raw_rpm, cutoff, fs)

            axs5[i].plot(t, raw_rpm, label=f"{rpm_labels[i]} (no filtered)", color='lightcoral', alpha=0.5)
            axs5[i].plot(t, filtered_rpm, label=f"{rpm_labels[i]} (filtered)", color='red')
            axs5[i].set_ylabel(rpm_labels[i])
            axs5[i].legend()
            axs5[i].grid(True)

        axs5[-1].set_xlabel('time (s)')
        fig5.suptitle('Motor\'s RPM')
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        fig5.savefig(os.path.join(save_path, "RPM_filtered.png"))

        v_labels = ['vx', 'vy', 'vz']
        w_labels = ['wx', 'wy', 'wz']

        # 6. Linear and angular velocity without filtering
        fig6, axs6 = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        for i in range(3):
            axs6[i, 0].plot(t, self.states[drone, 3 + i, :], label=f"{v_labels[i]} measured", color='red')
            axs6[i, 0].axhline(0, color='b', linestyle='--', label=f"{v_labels[i]} desired")
            axs6[i, 0].set_ylabel(f"{v_labels[i]} (m/s)")
            axs6[i, 0].legend()
            axs6[i, 0].grid(True)

            axs6[i, 1].plot(t, self.states[drone, 9 + i, :], label=f"{w_labels[i]} measured", color='red')
            axs6[i, 1].axhline(0, color='b', linestyle='--', label=f"{w_labels[i]} desired")
            axs6[i, 1].set_ylabel(f"{w_labels[i]} (rad/s)")
            axs6[i, 1].legend()
            axs6[i, 1].grid(True)

        axs6[2, 0].set_xlabel("time (s)")
        axs6[2, 1].set_xlabel("time (s)")
        fig6.suptitle("Desired vs Measured Linear and Angular Velocity")
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        fig6.savefig(os.path.join(save_path, "desired_vs_measured_linear_and_angular_velocity.png"))

        # 7. Angular velocity filtered
        fig7, axs7 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            w_raw = self.states[drone, 9 + i, :]
            w_filt = butter_lowpass_filter(w_raw, cutoff, fs)
            axs7[i].plot(t, w_raw, label=f"{w_labels[i]} sin filtrar", color='gray', alpha=0.5)
            axs7[i].plot(t, w_filt, label=f"{w_labels[i]} filtrado", color='red')
            axs7[i].axhline(0, color='b', linestyle='--', label=f"{w_labels[i]} desired")
            axs7[i].set_ylabel(f"{w_labels[i]} (rad/s)")
            axs7[i].legend()
            axs7[i].grid(True)
        axs7[2].set_xlabel("time (s)")
        fig7.suptitle("Desired vs Measured Angular Velocity")
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        fig7.savefig(os.path.join(save_path, "desired_vs_measured_angular_velocity_filtered.png"))

        # 8. Linear and angular velocity (both filtered)
        fig8, axs8 = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        for i in range(3):
            v_raw = self.states[drone, 3 + i, :]
            v_filt = butter_lowpass_filter(v_raw, cutoff, fs)
            axs8[i, 0].plot(t, v_filt, label=f"{v_labels[i]} filtered", color='red')
            axs8[i, 0].axhline(0, color='b', linestyle='--', label=f"{v_labels[i]} desired")
            axs8[i, 0].set_ylabel(f"{v_labels[i]} (m/s)")
            axs8[i, 0].legend()
            axs8[i, 0].grid(True)

            w_raw = self.states[drone, 9 + i, :]
            w_filt = butter_lowpass_filter(w_raw, cutoff, fs)
            axs8[i, 1].plot(t, w_filt, label=f"{w_labels[i]} filtered", color='red')
            axs8[i, 1].axhline(0, color='b', linestyle='--', label=f"{w_labels[i]} desired")
            axs8[i, 1].set_ylabel(f"{w_labels[i]} (rad/s)")
            axs8[i, 1].legend()
            axs8[i, 1].grid(True)

        axs8[2, 0].set_xlabel("time (s)")
        axs8[2, 1].set_xlabel("time (s)")
        fig8.suptitle("Desired vs Measured Linear and Angular Velocity")
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        fig8.savefig(os.path.join(save_path, "desired_vs_measured_linear_and_angular_velocity_filtered.png"))

        # 9. Linear velocity (unfiltered) and angular velocity (filtered)
        fig9, axs9 = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        for i in range(3):
            v_raw = self.states[drone, 3 + i, :]
            axs9[i, 0].plot(t, v_raw, label=f"{v_labels[i]} measured", color='red')
            axs9[i, 0].axhline(0, color='b', linestyle='--', label=f"{v_labels[i]} desired")
            axs9[i, 0].set_ylabel(f"{v_labels[i]} (m/s)")
            axs9[i, 0].legend()
            axs9[i, 0].grid(True)

            w_raw = self.states[drone, 9 + i, :]
            w_filt = butter_lowpass_filter(w_raw, cutoff, fs)
            axs9[i, 1].plot(t, w_filt, label=f"{w_labels[i]} filtered", color='red')
            axs9[i, 1].axhline(0, color='b', linestyle='--', label=f"{w_labels[i]} desired")
            axs9[i, 1].set_ylabel(f"{w_labels[i]} (rad/s)")
            axs9[i, 1].legend()
            axs9[i, 1].grid(True)

        axs9[2, 0].set_xlabel("time (s)")
        axs9[2, 1].set_xlabel("time (s)")
        fig9.suptitle("Desired vs Measured Linear and Angular Velocity")
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        fig9.savefig(os.path.join(save_path, "desired_vs_measured_linear_and_angular_velocity.png"))

        plt.show()
