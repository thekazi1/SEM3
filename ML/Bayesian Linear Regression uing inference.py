import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate data
X = np.linspace(0, 1, 20)
true_slope = 3
true_intercept = 1
y = true_slope * X + true_intercept + np.random.normal(scale=0.5, size=X.shape)

plt.scatter(X, y, label="Data")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Design matrix
X_ = np.vstack([X, np.ones_like(X)]).T

# Priors
sigma_prior = 10
mu_prior = np.array([0, 0])

# Likelihood noise
sigma_likelihood = 0.5

# Posterior covariance
X_T = X_.T
covariance_post = np.linalg.inv(
    X_T @ X_ / sigma_likelihood**2 + np.eye(2) / sigma_prior**2
)

# Posterior mean
mean_post = covariance_post @ (
    X_T @ y / sigma_likelihood**2 + mu_prior / sigma_prior**2
)

# Sample from posterior
num_samples = 1000
posterior_samples = np.random.multivariate_normal(mean_post, covariance_post, num_samples)

# Plot posterior histograms
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(posterior_samples[:, 0], bins=30, color='skyblue', edgecolor='black')
plt.title("Posterior Distribution of Slope")

plt.subplot(1, 2, 2)
plt.hist(posterior_samples[:, 1], bins=30, color='skyblue', edgecolor='black')
plt.title("Posterior Distribution of Intercept")

plt.tight_layout()
plt.show()

# Plot sampled regression lines
plt.scatter(X, y, label="Data")
for sample in posterior_samples:
    plt.plot(X, sample[0] * X + sample[1], color='red', alpha=0.05)

plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Final estimates
estimated_slope = mean_post[0]
estimated_intercept = mean_post[1]

print(f"Estimated Slope: {estimated_slope}")
print(f"Estimated Intercept: {estimated_intercept}")
