from math import atan

class YawController(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.min_speed = min_speed
        self.max_lat_accel = max_lat_accel

        self.min_angle = -max_steer_angle
        self.max_angle = max_steer_angle

        self.prev_steering = None
        self.steering = None


    def get_angle(self, radius):
        angle = atan(self.wheel_base / radius) * self.steer_ratio
        return max(self.min_angle, min(self.max_angle, angle))

    def get_steering(self, linear_velocity, angular_velocity, current_velocity):
        angular_velocity = current_velocity * angular_velocity / linear_velocity if abs(linear_velocity) > 0. else 0.

        if abs(current_velocity) > 0.1:
            max_yaw_rate = abs(self.max_lat_accel / current_velocity)
            angular_velocity = max(-max_yaw_rate, min(max_yaw_rate, angular_velocity))

        steering = self.get_angle(max(current_velocity, self.min_speed) / angular_velocity) if abs(angular_velocity) > 0. else 0.0

        if self.prev_steering:
            steer_diff = steering - self.prev_steering
            thresh = 0.001
            max_thresh = 0.01
            damp_val = 0.125
            if abs(steer_diff) > thresh:
                if abs(steer_diff) > max_thresh:
                    steering = 0.0
                else:
                    steering = self.prev_steering + (steer_diff * damp_val)

        self.prev_steering = self.steering
        self.steering = steering
        return self.steering
