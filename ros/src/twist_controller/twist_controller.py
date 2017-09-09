import pid
import lowpass

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, controller_rate, accel_limit, decel_limit, max_steer_angle, vehicle_mass, wheel_radius):

        self.controller_rate = controller_rate
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        self.vehicle_mass

        # TODO: find good parameters for PID controllers
        self.pid_throttle = pid.PID(kp=1.0, ki=0.0, kd=0.0, mn=-1, mx=1)
        self.pid_steer = pid.PID(kp=1.0, ki=0.0, kd=0.0, mn=-max_steer_angle, mx=max_steer_angle)

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

        steering_angle = self.pid_steer.step(error=cross_track_error, sample_time=sample_time)
        steering_angle_filtered = self.filter_steer.filt(steering_angle)

        throttle = self.pid_throttle.step(error=speed_error, sample_time=sample_time)
        throttle_filtered = self.filter_throttle.filt(throttle)

        throttle_out = max(0, throttle_filtered) # TODO: does acceleration limit needs to be enforced here?
        brake_out = - min(0, throttle_filtered) * self.decel_limit * self.vehicle_mass * wheel_radius
        steering_out = steering_angle_filtered

        return throttle_out, brake_out, steering_out

    def reset(self):
        """
        resets pid controllers
        """
        self.pid_throttle.reset()
        self.pid_steer.reset()


def test_controller():
    """
    simple test of functionality by creating Controller object and performing
    one control step
    """
    controller = Controller(controller_rate=30,
                            accel_limit=10,
                            decel_limit=-10,
                            max_steer_angle=0.25)

    controller.control(1, 0.1)

if __name__ == "__main__":
    test_controller()
