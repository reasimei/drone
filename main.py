import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, tan


class DroneControlSim:
    def __init__(self):
        self.sim_time = 10 #仿真总时间
        self.sim_step = 0.002 #仿真步长
        self.drone_states = np.zeros((int(self.sim_time / self.sim_step), 12)) #无人机状态12个
        self.time = np.zeros((int(self.sim_time / self.sim_step),)) #当前仿真时间
        self.rate_cmd = np.zeros((int(self.sim_time / self.sim_step), 3)) #期望角速率pqr 单位rad/s
        self.attitude_cmd = np.zeros((int(self.sim_time / self.sim_step), 3)) #期望姿态角
        self.velocity_cmd = np.zeros((int(self.sim_time / self.sim_step), 3)) #期望速度
        self.position_cmd = np.zeros((int(self.sim_time / self.sim_step), 3)) #期望位置
        self.pointer = 0 #用于跟踪当前仿真步数
        #转动惯量，质量，重力加速度
        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 4.00e-3
        self.m = 0.5
        self.g = 9.8
        self.I = np.array([[self.I_xx, .0, .0], [.0, self.I_yy, .0], [.0, .0, self.I_zz]])
        #PID控制相关参数
        self.ierror=[0,0,0]#误差积分
        self.perror=[0,0,0]#前一时刻误差
        self.kp=0.05
        self.ki=0.001
        self.kd=0



    def drone_dynamics(self, T, M): #根据无人机运动学模型更新状态
        # Input:
        # T: float Thrust
        # M: np.array (3,)  Moments in three axes
        # Output: np.array (12,) the derivative (dx) of the drone 

        x = self.drone_states[self.pointer, 0]
        y = self.drone_states[self.pointer, 1]
        z = self.drone_states[self.pointer, 2]
        vx = self.drone_states[self.pointer, 3]
        vy = self.drone_states[self.pointer, 4]
        vz = self.drone_states[self.pointer, 5]
        phi = self.drone_states[self.pointer, 6]
        theta = self.drone_states[self.pointer, 7]
        psi = self.drone_states[self.pointer, 8]
        p = self.drone_states[self.pointer, 9]
        q = self.drone_states[self.pointer, 10]
        r = self.drone_states[self.pointer, 11]

        R_d_angle = np.array([[1, tan(theta) * sin(phi), tan(theta) * cos(phi)], \
                              [0, cos(phi), -sin(phi)], \
                              [0, sin(phi) / cos(theta), cos(phi) / cos(theta)]])

        R_E_B = np.array([[cos(theta) * cos(psi), cos(theta) * sin(psi), -sin(theta)], \
                          [sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi),
                           sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi), sin(phi) * cos(theta)], \
                          [cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi),
                           cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi), cos(phi) * cos(theta)]])

        d_position = np.array([vx, vy, vz])
        d_velocity = np.array([.0, .0, self.g]) + R_E_B.transpose() @ np.array([.0, .0, T]) / self.m
        d_angle = R_d_angle @ np.array([p, q, r])
        d_q = np.linalg.inv(self.I) @ (M - np.cross(np.array([p, q, r]), self.I @ np.array([p, q, r])))

        dx = np.concatenate((d_position, d_velocity, d_angle, d_q))

        return dx

    def rate_controller(self, cmd): #输入期望角速率，输出控制力矩
        # Input: cmd np.array (3,) rate commands
        # Output: M np.array (3,) moments
        self.rate_cmd[self.pointer] = cmd
        p = self.drone_states[self.pointer, 9]
        q = self.drone_states[self.pointer, 10]
        r = self.drone_states[self.pointer, 11]
        rate_current = [p, q, r] #当前角速率
        self.kp = 0.01 #kp小一点，控制力矩小一点，慢过渡
        self.ki = 0.000
        self.kd = 0
        error = [desire-current for desire,current in zip(cmd, rate_current)] #计算偏差
        self.ierror = [ie+e*self.sim_step for ie,e in zip(self.ierror, error)] #误差积分
        derror = [e-pe for e,pe in zip(error, self.perror)] #误差微分
        M = [self.kp * e + self.ki * ie + self.kd * de for e,ie,de in zip(error,self.ierror,derror)] #pid
        self.perror = error
        return M
        #pass

    def attitude_controller(self, cmd):#输入期望姿态角，输出控制力矩
        # Input: cmd np.array (3,) attitude commands
        # Output: M np.array (3,) rate commands
        self.attitude_cmd[self.pointer] = cmd
        phi = self.drone_states[self.pointer, 6]
        theta = self.drone_states[self.pointer, 7]
        psi = self.drone_states[self.pointer, 8]
        attitude_current = [phi, theta, psi] #当前姿态角
        self.kp = 0.6
        self.ki = 0
        self.kd = 0.1
        error = [desire-current for desire,current in zip(cmd, attitude_current)] #计算偏差
        self.ierror = [ie+e*self.sim_step for ie,e in zip(self.ierror, error)] #误差积分
        derror = [e-pe for e,pe in zip(error, self.perror)] #误差微分
        rate = [self.kp * e + self.ki * ie + self.kd * de for e,ie,de in zip(error,self.ierror,derror)] #pid 期望角速率
        M = self.rate_controller(rate)
        self.perror = error
        return M
        #pass

    def velocity_controller(self, cmd):
        # Input: cmd np.array (3,) velocity commands
        # Output: M np.array (2,) phi and theta commands and thrust cmd
        pass

    def position_controller(self, cmd):
        # Input: cmd np.array (3,) position commands
        # Output: M np.array (3,) velocity commands
        pass

    def run(self):  # 开始仿真
        for self.pointer in range(self.drone_states.shape[0] - 1):  # 遍历仿真时间内每一个步长
            self.time[self.pointer] = self.pointer * self.sim_step  # 计算当前仿真时间
            thrust_cmd = -10  # 控制输入-推力和力矩 4.9/0.5=9.8 与重力平衡就悬停在空中了
            #M = np.zeros((3,))
            cmd = [0.3,0.2,0.1]# 输入期望角速率、姿态角 单位 rad/s、rad
            #M = self.rate_controller(cmd)  # 计算出控制力矩
            M = self.attitude_controller(cmd) #计算出控制力矩
            dx = self.drone_dynamics(thrust_cmd, M)  # 调用了 drone_dynamics 方法，根据当前飞行器状态和控制输入，计算了飞行器状态的变化率（即飞行器动力学模型）
            self.drone_states[self.pointer + 1] = self.drone_states[self.pointer] + dx * self.sim_step  # 将计算得到的飞行器状态的变化率乘以仿真步长，并加到当前时刻的飞行器状态上，从而得到下一个时间步长的飞行器状态。
        self.time[-1] = self.sim_time  # 循环结束后，最后一个时间步长的仿真时间被赋值为仿真总时间，以确保时间的连续性。


    def plot_states(self):
        fig1, ax1 = plt.subplots(4, 3)
        self.position_cmd[-1] = self.position_cmd[-2]
        ax1[0, 0].plot(self.time, self.drone_states[:, 0], label='real')
        ax1[0, 0].plot(self.time, self.position_cmd[:, 0], label='cmd')
        ax1[0, 0].set_ylabel('x[m]')
        ax1[0, 1].plot(self.time, self.drone_states[:, 1])
        ax1[0, 1].plot(self.time, self.position_cmd[:, 1])
        ax1[0, 1].set_ylabel('y[m]')
        ax1[0, 2].plot(self.time, self.drone_states[:, 2])
        ax1[0, 2].plot(self.time, self.position_cmd[:, 2])
        ax1[0, 2].set_ylabel('z[m]')
        ax1[0, 0].legend()

        self.velocity_cmd[-1] = self.velocity_cmd[-2]
        ax1[1, 0].plot(self.time, self.drone_states[:, 3])
        ax1[1, 0].plot(self.time, self.velocity_cmd[:, 0])
        ax1[1, 0].set_ylabel('vx[m/s]')
        ax1[1, 1].plot(self.time, self.drone_states[:, 4])
        ax1[1, 1].plot(self.time, self.velocity_cmd[:, 1])
        ax1[1, 1].set_ylabel('vy[m/s]')
        ax1[1, 2].plot(self.time, self.drone_states[:, 5])
        ax1[1, 2].plot(self.time, self.velocity_cmd[:, 2])
        ax1[1, 2].set_ylabel('vz[m/s]')

        self.attitude_cmd[-1] = self.attitude_cmd[-2]
        ax1[2, 0].plot(self.time, self.drone_states[:, 6])
        ax1[2, 0].plot(self.time, self.attitude_cmd[:, 0])
        ax1[2, 0].set_ylabel('phi[rad]')
        ax1[2, 1].plot(self.time, self.drone_states[:, 7])
        ax1[2, 1].plot(self.time, self.attitude_cmd[:, 1])
        ax1[2, 1].set_ylabel('theta[rad]')
        ax1[2, 2].plot(self.time, self.drone_states[:, 8])
        ax1[2, 2].plot(self.time, self.attitude_cmd[:, 2])
        ax1[2, 2].set_ylabel('psi[rad]')

        self.rate_cmd[-1] = self.rate_cmd[-2]
        ax1[3, 0].plot(self.time, self.drone_states[:, 9])
        ax1[3, 0].plot(self.time, self.rate_cmd[:, 0])
        ax1[3, 0].set_ylabel('p[rad/s]')
        ax1[3, 1].plot(self.time, self.drone_states[:, 10])
        ax1[3, 1].plot(self.time, self.rate_cmd[:, 1])
        ax1[3, 1].set_ylabel('q[rad/s]')
        ax1[3, 2].plot(self.time, self.drone_states[:, 11])
        ax1[3, 2].plot(self.time, self.rate_cmd[:, 2])
        ax1[3, 2].set_ylabel('r[rad/s]')


if __name__ == "__main__":
    drone = DroneControlSim()
    drone.run()
    drone.plot_states()
    plt.show()