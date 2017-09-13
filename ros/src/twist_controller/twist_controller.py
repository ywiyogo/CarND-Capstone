import pid
import lowpass
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, controller_rate, accel_limit, decel_limit, max_steer_angle, vehicle_mass, wheel_radius):

        self.controller_rate = controller_rate
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius

        # TODO: find good parameters for PID controllers
        self.pid_throttle = pid.PID(kp=0.8, ki=0.01, kd=0.1, mn=-1, mx=1)
        self.pid_steer = pid.PID(kp=0.5, ki=0.01, kd=0.2, mn=-max_steer_angle, mx=max_steer_angle)

        self.filter_throttle = lowpass.LowPassFilter(tau=0.1, ts=1)
        self.filter_steer = lowpass.LowPassFilter(tau=0.1, ts=1)

    def control(self, speed_error, cross_track_error):
        """
        calculates control signal
        :param speed_error:
        :param cross_track_error: The cross track error (cte) is the current y
                position of the vehicle
        :return: throttle [1], brake [Nm], steer [rad]
        """
        sample_time = self.controller_rate

        rospy.loginfo('Error speed: %s, Error cte: %s', speed_error, cross_track_error)
        steering_angle = self.pid_steer.step(error=cross_track_error, sample_time=sample_time)
        rospy.loginfo('Steering angle PID output: %s', steering_angle)
        steering_angle_filtered = self.filter_steer.filt(steering_angle)
        rospy.loginfo('Steering angle PID output filtered: %s', steering_angle_filtered)

        throttle = self.pid_throttle.step(error=speed_error, sample_time=sample_time)
        rospy.loginfo('Throttle PID output: %s', throttle)
        throttle_filtered = self.filter_throttle.filt(throttle)
        rospy.loginfo('Throttle PID output filtered %s', throttle_filtered)

        throttle_out = max(0, throttle_filtered) # TODO: does acceleration limit needs to be enforced here?
        brake_out = - min(0, throttle_filtered) * self.decel_limit * self.vehicle_mass * self.wheel_radius
        steering_out = steering_angle_filtered

        return throttle_out, brake_out, steering_out

    def reset(self):
        """
        resets pid controllers
        """
        self.pid_throttle.reset()
        self.filter_throttle.reset()
        self.pid_steer.reset()
        self.filter_steer.reset()


def test_controller():
    """
    simple test of functionality by creating Controller object and performing
    one control step
    """
    controller = Controller(controller_rate=30,
                            accel_limit=10,
                            decel_limit=-10,
                            max_steer_angle=0.25,
                            vehicle_mass=1000,
                            wheel_radius=0.3)

    controller.control(1, 0.1)

if __name__ == "__main__":
    test_controller()
