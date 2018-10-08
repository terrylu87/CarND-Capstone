import rospy
from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
from math import pi as PI

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                       accel_limit, wheel_radius, wheel_base, steer_ratio,
                       max_lat_accel, max_steer_angle):
        # TODO: Implement
        # self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        kp = 10
        ki = 0.0
        kd = 5
        mn = -max_steer_angle  # Min throttle
        mx = max_steer_angle # Max throttle
        self.yaw_controller = PID(kp, ki, kd, mn, mx)

        kp = 0.5
        ki = 0.0
        kd = 0.1
        mn = 0.  # Min throttle
        mx = 0.2 # Max throttle
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        tau = 0.5 # 1/(2pi*tau) = cutoff frequency
        ts = 0.02 # Sample time
        self.vel_lpf = LowPassFilter(tau, ts)
        tau = 0.3 # 1/(2pi*tau) = cutoff frequency
        ts = 0.02 # Sample time
        self.yaw_lpf = LowPassFilter(tau, ts)
    
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.steer_ratio = steer_ratio

        self.last_time = rospy.get_time()

    def safeangle(self, angle):
        return (angle+PI) % (2*PI) - PI

    def control(self, current_vel, dbw_enabled, linear_vel, current_angle, command_angle):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        # steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        yaw_error = self.safeangle(command_angle-current_angle)
        self.last_yaw = current_angle
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        steering = self.yaw_controller.step(yaw_error, sample_time)
        steering = self.yaw_lpf.filt(steering)

        current_vel = self.vel_lpf.filt(current_vel)
        vel_error = linear_vel - current_vel
        self.last_vel = current_vel
        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        # if car is trying to stop, at traffic light stop line, hold the car with brake
        if linear_vel == 0 and current_vel < 0.1:
            throttle = 0
            brake = 700

        elif throttle < .1 and vel_error < 0:
            throttle = 0
            decel = min(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius # Torque N*m

        return throttle, brake, steering
