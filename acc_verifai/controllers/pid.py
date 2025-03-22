import numpy as np
# import scenic.core.distributions import 
from collections import deque
from scipy import signal

class PID:
    """Longitudinal control using a PID to reach a target speed.

    Arguments:
        K_P: Proportional gain
        K_D: Derivative gain
        K_I: Integral gain
        dt: time step
    """
    def __init__(self, K_P=0.5, K_D=0.1, K_I=0.2, dt=0.1, min=-1, max=1, ie=0, tau=0, int_sat = 20):
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self.tau = tau
        self._error_buffer = deque(maxlen=2)
        self._error_buffer_filter = deque(maxlen=2)
        self.filtered_error = 0
        self._min = min
        self._max = max
        self.ie = ie
        self.int_sat = int_sat

    def run_step(self, speed_error):
        """Estimate the throttle/brake of the vehicle based on the PID equations.

        Arguments:
            speed_error: target speed minus current speed

        Returns:
            a signal between -1 and 1, with negative values indicating braking.
        """
        error = speed_error
        self._error_buffer.append(error)
        self.filtered_error = self.derivative_filter(self.filtered_error, error)
        self._error_buffer_filter.append(self.filtered_error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer_filter[-1] - self._error_buffer_filter[-2]) / self._dt
            self.ie += (self._error_buffer[-1] + self._error_buffer[-2]) * self._dt/2
            self.ie = np.clip(self.ie, -self.int_sat, self.int_sat)
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = self.ie

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * self.ie),
                    self._min, self._max)
    
    def derivative_filter(self, previous, current):
        if self.tau > 0:
            new = self._dt/self.tau * (-previous + current) + previous
        else:
            new = current
        return new