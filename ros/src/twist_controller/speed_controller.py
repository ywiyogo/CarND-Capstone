import pid
import lowpass
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class SpeedController(object):
    def __init__(self, controller_rate, accel_limit, decel_limit, brake_deadband, vehicle_mass, wheel_radius):

        self.controller_rate = controller_rate
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        self.brake_deadband = brake_deadband
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius

        # TODO: find good parameters for PID controllers
        self.pid_acceleration = pid.PID(kp=1, ki=0, kd=0, mn=self.decel_limit, mx=self.accel_limit)
        self.pid_throttle = pid.PID(kp=0.8, ki=0.01, kd=0.1, mn=-1.0, mx=1.0)

        self.filter_throttle = lowpass.LowPassFilter(tau=0.1, ts=1)

    def control(self, target_linear_velocity, current_linear_velocity, current_linear_acceleration):
        """
        calculates control signal
        :param speed_error:
        :param cross_track_error: The cross track error (cte) is the current y
                position of the vehicle
        :return: throttle [1], brake [Nm], steer [rad]
        """

        velocity_error = target_linear_velocity - current_linear_velocity

        sample_time = 1.0 / self.controller_rate # TODO: measure time

        rospy.loginfo('Error speed: %s', velocity_error)

        acceleration_cmd = self.pid_acceleration.step(error=velocity_error, sample_time=sample_time)
        rospy.loginfo('Acc PID output: %s', acceleration_cmd)

        acceleration_error = acceleration_cmd - current_linear_acceleration

        if (acceleration_cmd < self.brake_deadband):
            throttle_out = 0
            brake_out = -acceleration_cmd * self.vehicle_mass * self.wheel_radius
            self.pid_acceleration.reset()
        elif (acceleration_cmd < 0):
            throttle_out = 0
            brake_out = 0
            self.pid_acceleration.reset()
        else:
            throttle_out = self.pid_throttle.step(error=acceleration_error, sample_time=sample_time)
            rospy.loginfo('Throttle PID output %s', throttle_out)
            brake_out = 0

        #throttle_filtered = self.filter_throttle.filt(throttle_cmd)


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
