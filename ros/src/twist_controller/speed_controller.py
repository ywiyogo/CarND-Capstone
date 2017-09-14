import pid
import lowpass
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class SpeedController(object):
    def __init__(self, controller_rate, accel_limit, decel_limit, vehicle_mass, wheel_radius):

        self.controller_rate = controller_rate
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius

        # TODO: find good parameters for PID controllers
        self.pid_throttle = pid.PID(kp=0.8, ki=0.01, kd=0.1, mn=-1, mx=1)

        self.filter_throttle = lowpass.LowPassFilter(tau=0.1, ts=1)

    def control(self, speed_error):
        """
        calculates control signal
        :param speed_error:
        :param cross_track_error: The cross track error (cte) is the current y
                position of the vehicle
        :return: throttle [1], brake [Nm], steer [rad]
        """
        sample_time = self.controller_rate

        rospy.loginfo('Error speed: %s', speed_error)

        throttle = self.pid_throttle.step(error=speed_error, sample_time=sample_time)
        rospy.loginfo('Throttle PID output: %s', throttle)
        throttle_filtered = self.filter_throttle.filt(throttle)
        rospy.loginfo('Throttle PID output filtered %s', throttle_filtered)

        throttle_out = max(0, throttle_filtered) # TODO: does acceleration limit needs to be enforced here?
        brake_out = - min(0, throttle_filtered) * self.decel_limit * self.vehicle_mass * self.wheel_radius

        return throttle_out, brake_out

    def reset(self):
        """
        resets pid controllers
        """
        self.pid_throttle.reset()
        self.filter_throttle.reset()


def test_controller():
    """
    simple test of functionality by creating Controller object and performing
    one control step
    """
    controller = SpeedController(controller_rate=30,
                            accel_limit=10,
                            decel_limit=-10,
                            vehicle_mass=1000,
                            wheel_radius=0.3)

    controller.control(1)

if __name__ == "__main__":
    test_controller()
