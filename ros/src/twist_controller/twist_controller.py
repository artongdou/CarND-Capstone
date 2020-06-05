from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter as LPF
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel,
                 max_steer_angle, vehicle_mass, decel_limit, accel_limit, wheel_radius):

        self.wheel_radius = wheel_radius
        self.vehicle_mass = vehicle_mass
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.long_controller = PID(1.5, 0.07, 18, decel_limit, accel_limit)
        self.yaw_controller = YawController(
            wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        tau = 0.5  # cutoff freq = 1/(2*pi*tau)
        ts = 0.02  # sample time
        self.vel_lpf = LPF(tau, ts)
        self.last_time = rospy.get_time()

    def control(self, twist_cmd, curr_vel, dbw_enabled):
        '''
        @param twist_cmd - geometry_msgs/Twist
        @param curr_vel  - geometry_msgs/Twist
        '''
        if dbw_enabled:
            target_speed = twist_cmd.linear.x
            curr_speed = curr_vel.linear.x
            filt_speed = self.vel_lpf.filt(curr_speed)
            target_yaw_rate = twist_cmd.angular.z
            sample_time = rospy.get_time() - self.last_time

            accel = self.long_controller.step(
                target_speed - filt_speed, sample_time)

            steer = self.yaw_controller.get_steering(
                target_speed, target_yaw_rate, filt_speed)

            if target_speed == 0 and filt_speed < 0.5:
                # Vehicle is close to stop, apply brake and hold vehicle stationary
                throttle, brake, steer = 0, 700, 0
            elif accel >= 0:
                throttle, brake = accel / self.accel_limit, 0
            else:
                throttle, brake = 0, abs(accel)*self.vehicle_mass*self.wheel_radius

            return throttle, brake, steer
        else:
            self.long_controller.reset()
            throttle, brake, steer = 0, 0, 0

        self.last_time = rospy.get_time()
        return throttle, brake, steer
