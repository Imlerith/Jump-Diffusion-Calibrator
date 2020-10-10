import numpy as np
from scipy.stats import truncnorm, invgamma, binom, beta, norm
from basic_calibrator cimport BasicCalibrator


cdef class HestonCalibrator(BasicCalibrator):

    cdef:
        double alpha_prior
        double beta_prior
        double p_prior
        double psi_prior
        double theta_prior
        double sigma_sq_theta_prior
        double kappa_prior
        double sigma_sq_kappa_prior
        double alpha_star
        double mu
        double kappa
        double theta
        double omega
        double psi
        double[:, :] all_params_array_full
        double[:, :] all_params_array_no_burnin
        mystruct params_dict

    def __init__(self, alpha_prior, beta_prior, p_prior, psi_prior, theta_prior, sigma_sq_theta_prior,
                 kappa_prior, sigma_sq_kappa_prior, mu, kappa, theta, omega, psi, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # --- initialize prior parameters
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.p_prior = p_prior
        self.psi_prior = psi_prior
        self.theta_prior = theta_prior
        self.sigma_sq_theta_prior = sigma_sq_theta_prior
        self.kappa_prior = kappa_prior
        self.sigma_sq_kappa_prior = sigma_sq_kappa_prior
        self.alpha_star = self.T / 2 + self.alpha_prior
        # --- iniitalize posterior parameters
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.omega = omega
        self.psi = psi

    cpdef calibrate(self, int n_mcmc_steps, int burn_in, double rejection_rate):
        # ----- generate starting values for V using a truncated normal distribution
        #       (present-time as well as shifted backward and forward)
        V_t_array = np.array(truncnorm.rvs(a=0, b=np.inf, loc=0.0225, scale=0.005, size=self.T + 2),
                                            dtype=np.float64)
        cdef double[:] V_t_minus_1_array = np.array(np.roll(V_t_array, 1), dtype=np.float64)
        V_t_minus_1_array[0] = 0
        cdef double[:] V_t_plus_1_array = np.array(np.roll(V_t_array, -1), dtype=np.float64)
        V_t_plus_1_array[-1] = 0

        # --- create a padded version of Y for computation purposes
        cdef double[:] Y_t_array = np.array(np.append(0, np.append(self.returns, 0)), dtype=np.float64)
        cdef double[:] Y_t_plus_1_array = np.array(np.roll(np.asarray(Y_t_array), -1), dtype=np.float64)
        Y_t_plus_1_array[-1] = 0

        self.all_params_array_full = np.array(np.zeros((n_mcmc_steps, len(V_t_array) + 5)), dtype=np.float64)

        cdef double[:] all_params_0 = np.concatenate((np.array([self.mu, self.kappa, self.theta, self.psi, self.omega],
                                                      dtype=np.float64), V_t_array))
        cdef double[:] all_params_t = np.array(np.zeros((len(V_t_array) + 5,)), dtype=np.float64)
        self.all_params_array_full[0, :] = all_params_0

        cdef double[:] zero_array = np.zeros((len(self.returns),), dtype=np.float64)

        omega_alpha = self.alpha_star
        for iter in range(1, n_mcmc_steps):

            # ------- 1. Gibbs' sampling of model parameters -------
            # ----- start with the initialized parameters and update them using MCMC
            # (a) drift
            mu_mean = self.mu_star(self.psi, self.omega, self.kappa, self.theta, V_t_array, self.returns,
                                   zero_array, zero_array, self.delta_t, self.mu_prior, self.sigma_sq_mu_prior)
            mu_variance = self.sigma_sq_star(self.psi, self.omega, V_t_array, self.delta_t, self.sigma_sq_mu_prior)
            self.mu = np.random.normal(mu_mean, np.sqrt(mu_variance))

            # (b) Omega
            omega_beta = self.beta_star(V_t_array, self.returns, zero_array, zero_array, self.mu, self.delta_t,
                                        self.kappa, self.theta, self.beta_prior, self.p_prior, self.psi_prior)
            self.omega = invgamma.rvs(omega_alpha, scale=omega_beta)

            # (c) psi
            psi_mean = self.psi_star(self.returns, V_t_array, zero_array, zero_array, self.mu, self.delta_t,
                                     self.kappa, self.theta, self.p_prior, self.psi_prior)
            psi_vola = np.sqrt(self.sigma_sq_psi_star(self.returns, V_t_array, zero_array, zero_array,
                                                      self.mu, self.delta_t, self.p_prior, self.omega))
            self.psi = np.random.normal(psi_mean, psi_vola)

            # (d) theta
            theta_mean = self.theta_star(self.returns, V_t_array, zero_array, zero_array, self.mu, self.delta_t,
                                         self.psi, self.kappa, self.omega, self.theta_prior,
                                         self.sigma_sq_theta_prior)
            theta_vola = np.sqrt(self.sigma_sq_theta_star(V_t_array, self.delta_t, self.kappa,
                                                          self.omega, self.sigma_sq_theta_prior))
            self.theta = truncnorm.rvs((0 - theta_mean) / theta_vola, (5 - theta_mean) / theta_vola, loc=theta_mean,
                                        scale=theta_vola)

            # (e) kappa
            kappa_mean = self.kappa_star(self.returns, V_t_array, zero_array, zero_array, self.mu, self.delta_t,
                                         self.psi, self.theta, self.omega, self.kappa_prior,
                                         self.sigma_sq_kappa_prior)
            kappa_vola = np.sqrt(self.sigma_sq_kappa_star(V_t_array, self.delta_t, self.theta,
                                                          self.omega, self.sigma_sq_kappa_prior))
            self.kappa = truncnorm.rvs((0 - kappa_mean) / kappa_vola, (5 - kappa_mean) / kappa_vola, loc=kappa_mean,
                                        scale=kappa_vola)

            # ------- 2. Metropolis-Hastings' sampling of variance paths -------
            Y_and_V_arrays = zip(Y_t_array, Y_t_plus_1_array, V_t_minus_1_array, V_t_array, V_t_plus_1_array)
            V_t_array_new = list()
            for t, (Y_t, Y_t_plus_1, V_t_minus_1, V_t, V_t_plus_1) in enumerate(Y_and_V_arrays):

                # ----- generate a proposal value
                V_proposal = np.random.normal(V_t, rejection_rate)

                # ----- get density of V at the previous and proposed values of V
                if t == 0:
                    V_density_at_curr = self.state_space_target_dist_t_0(V_t, Y_t_plus_1, V_t_plus_1, 0.0, 0.0,
                                                                         self.delta_t, self.mu, self.omega, self.psi,
                                                                         self.kappa, self.theta)
                    V_density_at_prop = self.state_space_target_dist_t_0(V_proposal, Y_t_plus_1, V_t_plus_1, 0.0, 0.0,
                                                                         self.delta_t, self.mu, self.omega, self.psi,
                                                                         self.kappa, self.theta)
                elif t != 0 and t <= len(self.returns):
                    V_density_at_curr = self.state_space_target_dist_t_1_to_T(V_t, Y_t, 0.0, 0.0, Y_t_plus_1,
                                                                              V_t_plus_1, V_t_minus_1, 0.0, 0.0,
                                                                              self.delta_t, self.mu, self.omega,
                                                                              self.psi, self.kappa, self.theta)
                    V_density_at_prop = self.state_space_target_dist_t_1_to_T(V_proposal, Y_t, 0.0, 0.0, Y_t_plus_1,
                                                                              V_t_plus_1, V_t_minus_1, 0.0, 0.0,
                                                                              self.delta_t, self.mu, self.omega,
                                                                              self.psi, self.kappa, self.theta)
                else:
                    V_density_at_curr = self.state_space_target_dist_t_T_plus_1(V_t, Y_t, 0.0, 0.0, V_t_minus_1,
                                                                                self.delta_t, self.mu, self.omega,
                                                                                self.psi, self.kappa, self.theta)
                    V_density_at_prop = self.state_space_target_dist_t_T_plus_1(V_proposal, Y_t, 0.0, 0.0, V_t_minus_1,
                                                                                self.delta_t, self.mu, self.omega,
                                                                                self.psi, self.kappa, self.theta)

                # ----- estimate an acceptance probability for a given variance value
                # corr_factor = norm.pdf(V_t, loc=V_proposal, scale=sigma_N) / norm.pdf(V_proposal, loc=V_t, scale=sigma_N)
                accept_prob = min(V_density_at_prop / V_density_at_curr, 1)
                u = np.random.uniform(0, 1)
                if u < accept_prob:
                    V_t = V_proposal
                V_t_array_new.append(V_t)
            # ----- save the updated values
            V_t_array = np.array(V_t_array_new)
            V_t_minus_1_array = np.roll(V_t_array, 1)
            V_t_minus_1_array[0] = 0
            V_t_plus_1_array = np.roll(V_t_array, -1)
            V_t_plus_1_array[-1] = 0

            all_params_t = np.concatenate((np.array([self.mu, self.kappa, self.theta, self.psi, self.omega]),
                                                         V_t_array_new))
            self.all_params_array_full[iter, :] = all_params_t

        self.all_params_array_no_burnin = self.all_params_array_full[burn_in:, :]
        mu_final = np.mean(self.all_params_array_no_burnin[:, 0])
        kappa_final = np.mean(self.all_params_array_no_burnin[:, 1])
        theta_final = np.mean(self.all_params_array_no_burnin[:, 2])
        psi_final = np.mean(self.all_params_array_no_burnin[:, 3])
        omega_final = np.mean(self.all_params_array_no_burnin[:, 4])
        rho_final = np.sqrt(1 / (1 + omega_final / (psi_final ** 2)))
        volvol_final = psi_final / rho_final
        if volvol_final < 0:
            rho_final = -rho_final
            volvol_final = psi_final / rho_final
        self.params_dict = {"mu_final": mu_final, "kappa_final": kappa_final, "theta_final": theta_final,
                            "volvol_final": volvol_final, "rho_final": rho_final}

    def get_paths(self, s0=100, nsteps=2000, nsim=100, risk_neutral=False):
        mu = self.params_dict.get("mu_final")
        kappa = self.params_dict.get("kappa_final")
        theta = self.params_dict.get("theta_final")
        sigma = self.params_dict.get("volvol_final")
        rho = self.params_dict.get("rho_final")
        v0 = theta
        dt = 1 / nsteps
        simulated_paths = np.zeros([nsim, nsteps + 1])
        simulated_paths[:, 0] = s0
        simulated_volas = np.zeros([nsim, nsteps + 1])
        simulated_volas[:, 0] = v0

        # --- get randomness (correlated for each t=1,...,T, as corr(W_S, W_V) = rho)
        Z_V = np.random.normal(size=[nsim, nsteps + 1])
        Z_corr = np.random.normal(size=[nsim, nsteps + 1])
        Z_S = rho * Z_V + np.sqrt(1 - rho ** 2) * Z_corr

        # ----- generate paths
        for i in range(nsteps):
            # --- get the stochastic volatility component
            simulated_volas[:, i + 1] = simulated_volas[:, i] + kappa * (theta - simulated_volas[:, i]) * dt + \
                                        sigma * np.sqrt(simulated_volas[:, i]) * np.sqrt(dt) * Z_V[:, i + 1] + (
                                                0.5 ** 2) * (sigma ** 2) * dt * (Z_V[:, i + 1] ** 2 - 1)
            simulated_volas[:, i + 1] = list(map(lambda x: max(0, x), simulated_volas[:, i + 1]))

            # --- get drift with compensator
            if risk_neutral:
                drift = self.cost_of_carry - simulated_volas[:, i + 1] / 2
            else:
                drift = mu - simulated_volas[:, i + 1] / 2

            # --- get the total price dynamics
            simulated_paths[:, i + 1] = simulated_paths[:, i] * np.exp(
                drift * dt + np.sqrt(simulated_volas[:, i + 1] * dt) * Z_S[:, i + 1])
        return simulated_paths, simulated_volas

    @property
    def all_params_array_full(self):
        return np.asarray(self.all_params_array_full)

    @property
    def all_params_array_no_burnin(self):
        return np.asarray(self.all_params_array_no_burnin)

    @property
    def params_dict(self):
        return self.params_dict
