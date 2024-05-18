# -*- coding: utf-8 -*-
" " "
Created on Mon Apr 29 22:35:19 2024
@author: rjestanislao

import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pymc as pm
import seaborn as sns


def glm_mcmc_inference(df, iterations=5000):

    model = bmb.Model("y ~ x", df)

    trace = model.fit(
        draws=5000,
        tune=500,
        discard_tuned_samples=True,
        chains=1, 
        progressbar=True)
    return trace

def plot_glm_model(trace):
    pm.plot_trace(trace)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    beta_0 = 1.0
    beta_1 = 2.0

    start = 0
    stop = 1
    N = 100
    eps_mean = 0.0
    eps_sigma_sq = 0.5

    s = 42

    df = simulate_linear_data(
        start, stop, N, beta_0, beta_1, eps_mean, eps_sigma_sq
    )
    plot_simulated_data(df)

    trace = glm_mcmc_inference(df, iterations=5000)
    plot_glm_model(trace))