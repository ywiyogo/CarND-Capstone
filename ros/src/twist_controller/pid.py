import rospy

MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    def __init__(self, kp, ki, kd, k_anti_wind_up=0.5,
                 mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.k_anti_wind_up = k_anti_wind_up
        self.min = mn
        self.max = mx

        self.int_val = self.last_int_val = self.last_error = 0.
        self.last_wind_up = 0.

    def reset(self):
        self.int_val = 0.0
        self.last_wind_up = 0.
        self.last_int_val = 0.0

    def step(self, error, sample_time, verbose=False):
        self.last_int_val = self.int_val

        error_integral = error + self.k_anti_wind_up * self.last_wind_up
        integral = sample_time * error_integral + self.int_val

        derivative = (error - self.last_error) / sample_time;

        y = self.kp * error + self.ki * self.int_val + self.kd * derivative;
        val = max(self.min, min(y, self.max))

        self.last_wind_up = val - y

        self.int_val = integral

        self.last_error = error

        if verbose:
            rospy.loginfo('P: {}'.format(self.kp * error))
            rospy.loginfo('I: {}'.format(self.ki * self.int_val))
            rospy.loginfo('D: {}'.format(self.kd * derivative))


        return val
