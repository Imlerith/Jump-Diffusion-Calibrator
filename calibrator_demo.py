import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette='viridis')

import jumpdiffcalibrator as jdcal

# ----- load data
stock_data = pd.read_csv("aapl_data.csv")["Adj Close"].values

# ----- set parameters
s0 = 100
nsteps = 2000
nsim = 100
r = 0.05
q = 0.02

# ================================== Heston calibration====================================
# ----- calibrate parameters
n_mcmc_steps = 1000
burn_in = 500
heston_cal = jdcal.HestonCalibrator(price_series=stock_data, cost_of_carry=r - q)

start = time.time()
heston_cal.calibrate(n_mcmc_steps=n_mcmc_steps, burn_in=burn_in)
finish = time.time()
print(f"{(finish-start)/60} minutes elapsed")

# ----- get the calibrated parameters
all_params = heston_cal.params_dict
mu = all_params.get("mu_final")
kappa = all_params.get("kappa_final")
theta = all_params.get("theta_final")
sigma = all_params.get("volvol_final")
rho = all_params.get("rho_final")

# ----- simulate stock and vola trajectories
simulated_paths, simulated_volas = heston_cal.get_paths(s0=s0, nsteps=nsteps, nsim=nsim, risk_neutral=False)

# ----- get the figures
# --- (a) parameters' dynamics
offset = 150
burn_in_pos = burn_in - offset
param_paths = heston_cal.all_params_array_full[offset:, :]
fig, axes = plt.subplots(3, 2, figsize=(10, 12))
axes[0, 0].plot(param_paths[:, 0])
axes[0, 0].axvline(x=burn_in_pos, color="red", linewidth=3)
axes[0, 0].set_xlabel("$\\mu$")
axes[0, 1].plot(param_paths[:, 1])
axes[0, 1].axvline(x=burn_in_pos, color="red", linewidth=3)
axes[0, 1].set_xlabel("$\\kappa$")
axes[1, 0].plot(param_paths[:, 2])
axes[1, 0].axvline(x=burn_in_pos, color="red", linewidth=3)
axes[1, 0].set_xlabel("$\\theta$")
axes[1, 1].plot(param_paths[:, 3])
axes[1, 1].axvline(x=burn_in_pos, color="red", linewidth=3)
axes[1, 1].set_xlabel("$\\psi$")
axes[2, 0].plot(param_paths[:, 4])
axes[2, 0].axvline(x=burn_in_pos, color="red", linewidth=3)
axes[2, 0].set_xlabel("$\\omega$")
axes[2, 1].remove()
plt.suptitle('Posterior dynamics of parameters in Heston model (burn-in cutoff in red)')
plt.subplots_adjust(wspace=None, hspace=0.3)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- (b) price paths
fig, ax = plt.subplots(figsize=(12, 6))
days = np.linspace(0, 1, nsteps + 1) * nsteps
bates_prices = ax.plot(days, simulated_paths.transpose(), linewidth=1)
ax.set_title("Monte-Carlo simulated stock price paths in Heston model:\n$S_0$ = {}, $r-q$ = {}, "
             "$\\sigma$ = {}, $V_0$ = {}, $\\theta$ = {}, $\\kappa$ = {}, $\\rho$ = {}, "
             "Nsteps = {}, Nsim = {}".format(*list(map(lambda x: round(x, 3), [s0, r - q, np.round(sigma, 3),
                                                                               np.round(theta, 3), np.round(theta, 3),
                                                                               np.round(kappa, 3), np.round(rho, 3),
                                                                               nsteps, nsim]))))
ax.set_xlabel('Time (days)')
ax.set_ylabel('Stock price')

# --- (c) price volas
fig, ax = plt.subplots(figsize=(12, 6))
days = np.linspace(0, 1, nsteps + 1) * nsteps
bates_volas = ax.plot(days, simulated_volas.transpose(), linewidth=1)
ax.set_title("Monte-Carlo simulated vola paths in Heston model:\n$S_0$ = {}, $r$ = {}, "
             "$\\sigma$ = {}, $V_0$ = {}, $\\theta$ = {}, $\\kappa$ = {}, $\\rho$ = {}, "
             "Nsteps = {}, Nsim = {}".format(*list(map(lambda x: round(x, 3), [s0, r - q, np.round(sigma, 3),
                                                                               np.round(theta, 3), np.round(theta, 3),
                                                                               np.round(kappa, 3), np.round(rho, 3),
                                                                               nsteps, nsim]))))
ax.set_xlabel('Time (days)')
ax.set_ylabel('Variance process')

# ================================== Bates calibration ====================================
# ----- calibrate parameters
n_mcmc_steps = 10000
burn_in = 5000
bates_cal = jdcal.BatesCalibrator(price_series=stock_data, cost_of_carry=r - q)

start = time.time()
bates_cal.calibrate(n_mcmc_steps=n_mcmc_steps, burn_in=burn_in)
finish = time.time()
print(f"{(finish-start)/60} minutes elapsed")

# ----- get the calibrated parameters
all_params = bates_cal.params_dict
mu = all_params.get("mu_final")
kappa = all_params.get("kappa_final")
theta = all_params.get("theta_final")
sigma = all_params.get("volvol_final")
rho = all_params.get("rho_final")
mu_s = all_params.get("mu_s_final")
sigma_s = np.sqrt(all_params.get("sigma_sq_s_final"))
lambda_d = all_params.get("lambda_d_final")

# ----- simulate stock and vola trajectories
simulated_paths, simulated_volas = bates_cal.get_paths(s0=s0, nsteps=nsteps, nsim=nsim, risk_neutral=True)

# ----- get the figures
# --- (a) parameters' dynamics
offset = 200
burn_in_pos = burn_in - offset
param_paths = bates_cal.all_params_array_full[offset:, :]
fig, axes = plt.subplots(4, 2, figsize=(10, 12))
axes[0, 0].plot(param_paths[:, 0])
axes[0, 0].axvline(x=burn_in_pos, color="red", linewidth=3)
axes[0, 0].set_xlabel("$\\mu$")
axes[0, 1].plot(param_paths[:, 1])
axes[0, 1].axvline(x=burn_in_pos, color="red", linewidth=3)
axes[0, 1].set_xlabel("$\\kappa$")
axes[1, 0].plot(param_paths[:, 2])
axes[1, 0].axvline(x=burn_in_pos, color="red", linewidth=3)
axes[1, 0].set_xlabel("$\\theta$")
axes[1, 1].plot(param_paths[:, 3])
axes[1, 1].axvline(x=burn_in_pos, color="red", linewidth=3)
axes[1, 1].set_xlabel("$\\psi$")
axes[2, 0].plot(param_paths[:, 4])
axes[2, 0].axvline(x=burn_in_pos, color="red", linewidth=3)
axes[2, 0].set_xlabel("$\\omega$")
axes[2, 1].plot(param_paths[:, 5])
axes[2, 1].axvline(x=burn_in_pos, color="red", linewidth=3)
axes[2, 1].set_xlabel("$\\mu_Y$")
axes[3, 0].plot(np.sqrt(param_paths[:, 6]))
axes[3, 0].axvline(x=burn_in_pos, color="red", linewidth=3)
axes[3, 0].set_xlabel("$\\sigma_Y$")
axes[3, 1].plot(param_paths[:, 7])
axes[3, 1].axvline(x=burn_in_pos, color="red", linewidth=3)
axes[3, 1].set_xlabel("$\\lambda$")
plt.suptitle('Posterior dynamics of parameters in Bates model (burn-in cutoff in red)')
plt.subplots_adjust(wspace=None, hspace=0.3)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- (b) prices
fig, ax = plt.subplots(figsize=(12, 6))
days = np.linspace(0, 1, nsteps + 1) * nsteps
bates_prices = ax.plot(days, simulated_paths.transpose(), linewidth=1)
ax.set_title("Monte-Carlo simulated stock price paths in Bates' model: \n$S_0$ = {}, $r$ = {}, $q$ = {}, "
             "$\\sigma$ = {}, $\\mu_Y$ = {}, $\\sigma_Y$ = {}, $\\lambda$ = {}, $V_0$ = {}, $\\theta$ = {}, "
             "$\\kappa$ = {}, $\\rho$ = {}, Nsteps = {}, Nsim = {}"
             .format(s0, r, q, np.round(sigma, 3), np.round(mu_s, 3), np.round(sigma_s, 3), np.round(lambda_d, 3),
                     np.round(theta, 3), np.round(theta, 3), np.round(kappa, 3), np.round(rho, 3), nsteps, nsim))
ax.set_xlabel('Time (days)')
ax.set_ylabel('Stock price')

# --- (c) volas
fig, ax = plt.subplots(figsize=(12, 6))
days = np.linspace(0, 1, nsteps + 1) * nsteps
bates_volas = ax.plot(days, simulated_volas.transpose(), linewidth=1)
ax.set_title("Monte-Carlo simulated vola paths in Bates' model: \n$S_0$ = {}, $r$ = {}, $q$ = {}, "
             "$\\sigma$ = {}, $\\mu_Y$ = {}, $\\sigma_Y$ = {}, $\\lambda$ = {}, $V_0$ = {}, "
             "$\\theta$ = {}, $\\kappa$ = {}, $\\rho$ = {}, Nsteps = {}, Nsim = {}"
             .format(s0, r, q, np.round(sigma, 3), np.round(mu_s, 3), np.round(sigma_s, 3), np.round(lambda_d, 3),
                     np.round(theta, 3), np.round(theta, 3), np.round(kappa, 3), np.round(rho, 3), nsteps, nsim))
ax.set_xlabel('Time (days)')
ax.set_ylabel('Variance process')
