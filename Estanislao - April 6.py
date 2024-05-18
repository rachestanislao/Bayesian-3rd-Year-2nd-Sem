import numpy as np
import matplotlib.pyplot as plt

prior_lower_bound = 0
prior_upper_bound = 10

observed_data = np.random.uniform(2, 8, 20)

plt.hist(observed_data, bins=10, density=True, alpha=0.6, color='g', label='Observed Data')

posterior_lower_bound = np.min(observed_data)
posterior_upper_bound = np.max(observed_data)

X = np.linspace(0, 10, 1000)
prior_pdf = np.ones_like(X) / (prior_upper_bound - prior_lower_bound)
posterior_pdf = np.where((X >= posterior_lower_bound) & (X <= posterior_upper_bound), 1 / (posterior_upper_bound - posterior_lower_bound), 0)
plt.plot(X, prior_pdf, color='b', linestyle='-', label='Prior')
plt.plot(X, posterior_pdf, color='r', linestyle='-', label='Posterior')

plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Prior and Posterior Distributions')
plt.legend()
plt.grid(True)
plt.show()