GAS_DENSITY = 2.858
ONE_MPH = 0.44704
import pid
import lowpass


class Controller(object):
    def __init__(self, accel_limit, decel_limit, max_steer_angle):
        # TODO: find good parameters for PID controllers
        self.pid_throttle = pid.PID(kp=1.0, ki=0.0, kd=0.0, mn=decel_limit, mx=accel_limit)
        self.pid_steer = pid.PID(kp=1.0, ki=0.0, kd=0.0, mn=-max_steer_angle, mx=max_steer_angle)

        self.filter_throttle = lowpass.LowPassFilter(tau=0.1, ts=1)
        self.filter_steer = lowpass.LowPassFilter(tau=0.1, ts=1)

    def control(self, speed_error, cross_track_error):
        """

        :param speed_error:
        :param cross_track_error: The cross track error (cte) is the current y position of the vehicle
        :return: throttle, brake, steer
        """
        sample_time = 0.1 # TODO: adjust to real system cycle time

        steering_angle = self.pid_steer.step(error=cross_track_error, sample_time=sample_time)
        steering_angle_filtered = self.filter_steer.filt(steering_angle)

        throttle = self.pid_throttle.step(error=speed_error, sample_time=sample_time)
        throttle_filtered = self.filter_throttle.filt(throttle)

        throttle_out = max(0, throttle_filtered)
        brake_out = min(0, throttle_filtered)
        steering_out = steering_angle_filtered

        return throttle_out, brake_out, steering_out

    def reset(self):
        self.pid_throttle.reset()
        self.pid_brake.reset()
        self.pid_steer.reset()


def test_controller():
    controller = Controller(accel_limit=10, decel_limit=-10, max_steer_angle=0.25)
    controller.control(1, 0.1)

if __name__ == "__main__":
    test_controller()
