from abc import ABC, abstractmethod
import numpy as np


class BasicCalibrator(ABC):

    def __init__(self, price_series, cost_of_carry: float = 0.03, mu_prior: float = 0.0,
                 sigma_sq_mu_prior: float = 1.0, delta_t: float = 1.0):
        self.mu_prior = mu_prior
        self.sigma_sq_mu_prior = sigma_sq_mu_prior

        self.returns = np.diff(np.log(price_series))  # "Y" is "returns" here
        self.s0 = price_series[0]
        self.T = len(self.returns)
        self.delta_t = delta_t
        self.cost_of_carry = cost_of_carry

    @abstractmethod
    def calibrate(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_paths(self, *args, **kwargs):
        pass
