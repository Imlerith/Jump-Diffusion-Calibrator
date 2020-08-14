import numpy as np


class ConditionalPosteriorHestonMixin:

    @staticmethod
    def mu_star(psi, omega, kappa, theta, V, Y, Z, B, dt, mu_prior, sigma_sq_mu_prior):
        """ Posterior mean for the drift parameter"""
        numerator = sum((omega + psi ** 2) * (Y + 0.5 * V[:-2] * dt - Z * B) / (omega * V[:-2])) - \
                    sum(psi * (V[1:-1] - kappa * theta * dt - (1 - kappa * dt) * V[:-2]) / (omega * V[:-2])) \
                    + mu_prior / sigma_sq_mu_prior
        denominator = dt * sum((omega + psi ** 2) / (omega * V[:-2])) + 1 / sigma_sq_mu_prior
        return numerator / denominator

    @staticmethod
    def sigma_sq_star(psi, omega, V, dt, sigma_prior):
        """ Posterior variance for the drift parameter"""
        numerator = 1
        denominator = dt * sum((omega + psi ** 2) / (omega * V[:-2])) + 1 / (sigma_prior ** 2)
        return numerator / denominator

    @staticmethod
    def get_eps_s(V, Y, Z, B, mu, dt):
        return (Y - mu * dt + 0.5 * V[:-2] * dt - Z * B) / np.sqrt(V[:-2] * dt)

    @staticmethod
    def get_eps_v(V, dt, kappa, theta):
        return (V[1:-1] - kappa * theta * dt - (1 - kappa * dt) * V[:-2]) / np.sqrt(V[:-2] * dt)

    @classmethod
    def beta_star(cls, V, Y, Z, B, mu, dt, kappa, theta, beta_prior, p_prior, psi_prior):
        """ Posterior beta parameter for Omega which is
        used to parameterize the variance of variance and
        the correlation of the stock and variance processes"""
        eps_S = cls.get_eps_s(V, Y, Z, B, mu, dt)
        eps_V = cls.get_eps_v(V, dt, kappa, theta)
        result = beta_prior + 0.5 * sum(eps_V ** 2) + 0.5 * p_prior * psi_prior ** 2 - \
                 0.5 * ((p_prior * psi_prior + sum(eps_S * eps_V)) ** 2 / (p_prior + sum(eps_S ** 2)))
        return result

    @classmethod
    def psi_star(cls, Y, V, Z, B, mu, dt, kappa, theta, p_prior, psi_prior):
        """ Posterior mean parameter for psi which is also
        used to parameterize the variance of variance and
        the correlation of the stock and variance processes """
        eps_S = cls.get_eps_s(V, Y, Z, B, mu, dt)
        eps_V = cls.get_eps_v(V, dt, kappa, theta)
        result = (p_prior * psi_prior + sum(eps_S * eps_V)) / (p_prior + sum(eps_S ** 2))
        return result

    @classmethod
    def sigma_sq_psi_star(cls, Y, V, Z, B, mu, dt, p_prior, omega):
        """ Posterior variance parameter for psi which is used
        to parameterize the variance of variance and
        the correlation of the stock and variance processes """
        eps_S = cls.get_eps_s(V, Y, Z, B, mu, dt)
        result = omega / (p_prior + sum(eps_S ** 2))
        return result

    @staticmethod
    def theta_star(Y, V, Z, B, mu, dt, psi, kappa, omega, theta_prior, sigma_sq_theta_prior):
        """ Posterior mean parameter for the mean reversion parameter for
        the variance process """
        numerator = sum(kappa * (V[1:-1] - (1 - kappa * dt) * V[:-2]) / (omega * V[:-2])) - \
                    sum(psi * (Y - mu * dt + 0.5 * V[:-2] * dt - Z * B) * kappa / (omega * V[:-2]) +
                        theta_prior / sigma_sq_theta_prior)
        denominator = dt * sum(kappa ** 2 / (omega * V[:-2])) + 1 / sigma_sq_theta_prior
        theta = numerator / denominator
        return theta

    @staticmethod
    def sigma_sq_theta_star(V, dt, kappa, omega, sigma_sq_theta_prior):
        """ Posterior variance parameter for the mean reversion parameter for
        the variance process """
        denominator = dt * sum(kappa ** 2 / (omega * V[:-2])) + 1 / sigma_sq_theta_prior
        return 1 / denominator

    @staticmethod
    def kappa_star(Y, V, Z, B, mu, dt, psi, theta, omega, kappa_prior, sigma_sq_kappa_prior):
        """ Posterior mean parameter for the mean reversion rate parameter for
        the variance process """
        numerator = sum((theta - V[1:-1]) * (V[1:-1] - V[:-2]) / (omega * V[:-2])) - \
                    sum(psi * (Y - mu * dt + 0.5 * V[:-2] * dt - Z * B) * (theta - V[:-2]) / (omega * V[:-2])) + \
                    kappa_prior / sigma_sq_kappa_prior
        denominator = dt * sum((V[:-2] - theta) ** 2 / (omega * V[:-2])) + 1 / sigma_sq_kappa_prior
        return numerator / denominator

    @staticmethod
    def sigma_sq_kappa_star(V, dt, theta, omega, sigma_sq_kappa_prior):
        """ Posterior variance parameter for the mean reversion rate parameter for
        the variance process """
        denominator = dt * sum((V[:-2] - theta) ** 2 / (omega * V[:-2])) + 1 / sigma_sq_kappa_prior
        return 1 / denominator

    @staticmethod
    def mu_s_star(psi, omega, kappa, theta, V_t_minus_1, V_t, Y_t, mu, dt, mu_s, sigma_sq_s):
        """ Posterior mean for the jump size """
        numerator = ((omega + psi ** 2) * (Y_t + 0.5 * V_t_minus_1 * dt - mu * dt) / (omega * V_t_minus_1 * dt)) - \
                    (psi * (V_t - kappa * theta * dt - (1 - kappa * dt) * V_t_minus_1) / (omega * V_t_minus_1 * dt)) \
                    + mu_s / sigma_sq_s
        denominator = (omega + psi ** 2) / (omega * V_t_minus_1 * dt) + 1 / sigma_sq_s
        return numerator / denominator

    @staticmethod
    def sigma_sq_s_star(psi, omega, V_t_minus_1, dt, sigma_sq_s):
        """ Posterior variance for the jump size """
        denominator = (omega + psi ** 2) / (omega * V_t_minus_1 * dt) + 1 / sigma_sq_s
        return 1 / denominator

    @staticmethod
    def mu_m_s_star(S_0, sigma_sq_s, T, Z):
        numerator = sum(Z / sigma_sq_s)
        denominator = 1 / S_0 + T / sigma_sq_s
        return numerator / denominator

    @staticmethod
    def sigma_sq_m_s_star(S_0, sigma_sq_s, T):
        denominator = 1 / S_0 + T / sigma_sq_s
        return 1 / denominator

    @staticmethod
    def get_p_star(psi, omega, kappa, theta, V_t_minus_1, V_t, Y_t, Z_t, mu_drift, delta_t, lambda_d):
        A = ((omega + psi ** 2) * (
                Z_t ** 2 - 2 * Z_t * (Y_t - mu_drift * delta_t + 0.5 * V_t_minus_1 * delta_t)) + 2 * psi * (
                     V_t - kappa * theta * delta_t - (1 - kappa * delta_t) * V_t_minus_1) * Z_t) / (
                    omega * V_t_minus_1 * delta_t)
        denominator = (1 - lambda_d) * np.exp(0.5 * A) / lambda_d + 1
        return 1 / denominator

    @staticmethod
    def state_space_target_dist_term_1(V_proposed_or_current, Y_t_plus_1, Z_t_plus_1, B_t_plus_1,
                                       dt, mu, omega, psi):
        return (-1 / (2 * omega)) * (((omega + psi ** 2) * (
                0.5 * V_proposed_or_current * dt + Y_t_plus_1 - Z_t_plus_1 * B_t_plus_1 - mu * dt) ** 2) / (
                                             V_proposed_or_current * dt))

    @staticmethod
    def state_space_target_dist_term_2(V_proposed_or_current, Y_t_plus_1, V_t_plus_1, Z_t_plus_1, B_t_plus_1,
                                       dt, mu, omega, psi, kappa, theta):
        return (-1 / (2 * omega)) * (
                (-2 * psi * (0.5 * V_proposed_or_current * dt + Y_t_plus_1 - Z_t_plus_1 * B_t_plus_1 -
                             mu * dt) * (
                         (kappa * dt - 1) * V_proposed_or_current - kappa * theta * dt + V_t_plus_1)) / (
                        V_proposed_or_current * dt))

    @staticmethod
    def state_space_target_dist_term_3(V_proposed_or_current, V_t_plus_1, dt, omega, kappa, theta):
        return (-1 / (2 * omega)) * (
                ((kappa * dt - 1) * V_proposed_or_current - kappa * theta * dt + V_t_plus_1) ** 2 / (
                V_proposed_or_current * dt))

    @staticmethod
    def state_space_target_dist_term_4(V_proposed_or_current, Y_t, Z_t, B_t, V_t_minus_1,
                                       dt, mu, omega, psi, kappa, theta):
        return (-1 / (2 * omega)) * (
                -2 * psi * (Y_t - Z_t * B_t - mu * dt + 0.5 * V_t_minus_1 * dt) * (V_proposed_or_current -
                                                                                   kappa * theta * dt - (
                                                                                           1 - kappa * dt) * V_t_minus_1) / (
                        V_t_minus_1 * dt))

    @staticmethod
    def state_space_target_dist_term_5(V_proposed_or_current, V_t_minus_1, dt, omega, kappa, theta):
        return (-1 / (2 * omega)) * (
                (V_proposed_or_current - kappa * theta * dt - (1 - kappa * dt) * V_t_minus_1) ** 2 / (V_t_minus_1 * dt))

    @classmethod
    def state_space_target_dist_t_0(cls, V_proposed_or_current, Y_t_plus_1, V_t_plus_1, Z_t_plus_1, B_t_plus_1,
                                    dt, mu, omega, psi, kappa, theta):
        """ Formula for the target distribution of the state space """
        multiplier = 1 / (V_proposed_or_current * dt)
        term_1 = cls.state_space_target_dist_term_1(V_proposed_or_current, Y_t_plus_1, Z_t_plus_1, B_t_plus_1,
                                                    dt, mu, omega, psi)
        term_2 = cls.state_space_target_dist_term_2(V_proposed_or_current, Y_t_plus_1, V_t_plus_1, Z_t_plus_1,
                                                    B_t_plus_1,
                                                    dt, mu, omega, psi, kappa, theta)
        term_3 = cls.state_space_target_dist_term_3(V_proposed_or_current, V_t_plus_1, dt, omega, kappa, theta)
        return multiplier * np.exp(term_1 + term_2 + term_3)

    @classmethod
    def state_space_target_dist_t_1_to_T(cls, V_proposed_or_current, Y_t, Z_t, B_t, Y_t_plus_1, V_t_plus_1, V_t_minus_1,
                                         Z_t_plus_1, B_t_plus_1, dt, mu, omega, psi, kappa, theta):
        """ Formula for the target distribution of the state space """
        multiplier = 1 / (V_proposed_or_current * dt)
        term_1 = cls.state_space_target_dist_term_1(V_proposed_or_current, Y_t_plus_1, Z_t_plus_1, B_t_plus_1,
                                                    dt, mu, omega, psi)
        term_2 = cls.state_space_target_dist_term_2(V_proposed_or_current, Y_t_plus_1, V_t_plus_1, Z_t_plus_1,
                                                    B_t_plus_1,
                                                    dt, mu, omega, psi, kappa, theta)
        term_3 = cls.state_space_target_dist_term_3(V_proposed_or_current, V_t_plus_1, dt, omega, kappa, theta)
        term_4 = cls.state_space_target_dist_term_4(V_proposed_or_current, Y_t, Z_t, B_t, V_t_minus_1,
                                                    dt, mu, omega, psi, kappa, theta)
        term_5 = cls.state_space_target_dist_term_5(V_proposed_or_current, V_t_minus_1, dt, omega, kappa, theta)
        return multiplier * np.exp(term_1 + term_2 + term_3 + term_4 + term_5)

    @classmethod
    def state_space_target_dist_t_T_plus_1(cls, V_proposed_or_current, Y_t, Z_t, B_t, V_t_minus_1, dt,
                                           mu, omega, psi, kappa, theta):
        """ Formula for the target distribution of the state space """
        multiplier = 1 / (V_proposed_or_current * dt)
        term_4 = cls.state_space_target_dist_term_4(V_proposed_or_current, Y_t, Z_t, B_t, V_t_minus_1,
                                                    dt, mu, omega, psi, kappa, theta)
        term_5 = cls.state_space_target_dist_term_5(V_proposed_or_current, V_t_minus_1, dt, omega, kappa, theta)
        return multiplier * np.exp(term_4 + term_5)


class LazyProperty:
    """A descriptor class to evaluate properties lazily"""

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


def lazy_property(func):
    name = '_lazy_' + func.__name__

    @property
    def lazy(self):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            value = func(self)
            setattr(self, name, value)
            return value

    return lazy


def log_exception(level, default_result=None):
    # logger with default argument
    def log_internal(func):
        lgr = logging.getLogger("my_application")
        lgr.setLevel(level)
        # fh = logging.FileHandler("{}.log".format(func.__name__))
        fh = logging.FileHandler("my_logger.log")
        fh.setLevel(level)
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        if lgr.hasHandlers():
            lgr.handlers.clear()
        lgr.addHandler(fh)

        @wraps(func)
        def wrapper(*a, **kw):
            lgr.log(level, "Ran with args: {} and kwargs: {}".format(a, kw))
            try:
                return func(*a, **kw)
            except Exception as e:
                err = "There was an exception in  "
                err += func.__name__
                lgr.exception(err)
                lgr.exception(e)
                return default_result

        return wrapper

    return log_internal
