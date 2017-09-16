import pid
import lowpass
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class TwistController(object):
    def __init__(self, controller_rate, max_steer_angle):

        self.controller_rate = controller_rate

        # TODO: find good parameters for PID controllers
        self.pid_steer = pid.PID(kp=0.05, ki=0.002, kd=0.000,
                                 mn=-max_steer_angle, mx=max_steer_angle)

        self.filter_steer = lowpass.LowPassFilter(tau=0.1, ts=1)

    def control(self, cross_track_error):
        """
        calculates control signal
        :param speed_error:
        :param cross_track_error: The cross track error (cte) is the current y
                position of the vehicle
        :return: throttle [1], brake [Nm], steer [rad]
        """
        sample_time = 1. / self.controller_rate

        rospy.logwarn('Error cte: %s', cross_track_error)
        steering_angle = self.pid_steer.step(error=cross_track_error,
                                             sample_time=sample_time)
        rospy.logwarn('Steering angle PID output: %s', steering_angle)
        # steering_angle_filtered = self.filter_steer.filt(steering_angle)
        # rospy.loginfo('Steering angle PID output filtered: %s', steering_angle_filtered)

        steering_out = steering_angle # steering_angle_filtered

        return steering_out

    def reset(self):
        """
        resets pid controllers
        """
        rospy.loginfo('reset', steering_angle)

        self.pid_steer.reset()
        self.filter_steer.reset()


def test_controller():
    """
    simple test of functionality by creating Controller object and performing
    one control step
    """
    controller = TwistController(controller_rate=30,
                                 max_steer_angle=0.25)

    controller.control(0.1)

if __name__ == "__main__":
    test_controller()
