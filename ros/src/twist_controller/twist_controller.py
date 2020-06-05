from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter as LPF

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle, rate, vehicle_mass, decel_limit, accel_limit):
        self.rate = rate
        self.vehicle_mass = vehicle_mass
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.long_controller = PID(1.5, 0.07, 18, decel_limit, accel_limit)
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

    def control(self, twist_cmd, curr_vel, dbw_enabled):
        '''
        @param twist_cmd - geometry_msgs/Twist
        @param curr_vel  - geometry_msgs/Twist
        '''
        if dbw_enabled:
            target_speed = twist_cmd.linear.x
            curr_speed = curr_vel.linear.x
            target_yaw_rate = twist_cmd.angular.z
            curr_yaw_rate = curr_vel.angular.z

            accel = self.long_controller.step(target_speed - curr_speed, self.rate)

            steer = self.yaw_controller.get_steering(target_speed, target_yaw_rate, curr_speed)

            if accel >= 0:
                throttle, brake = accel / self.accel_limit, 0
            else:
                throttle, brake = 0, abs(accel)*self.vehicle_mass
        else:
            self.long_controller.reset()
            throttle, brake, steer = 0, 0, 0


        return throttle, brake, steer
