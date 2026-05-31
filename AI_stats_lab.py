import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor
)


def generate_clean_data():
    X, y, coef = make_regression(
        n_samples=500,
        n_features=1,
        n_informative=1,
        noise=20,
        coef=True,
        random_state=42
    )

    return X, y, float(coef)


def add_outliers(X, y):
    rng = np.random.RandomState(42)

    X_out = X.copy()
    y_out = y.copy()

    X_out[:25] = 10 + 0.75 * rng.randn(25, 1)
    y_out[:25] = -15 + 20 * rng.randn(25)

    return X_out, y_out


def plot_dataset_with_outliers(X, y):
    fig, ax = plt.subplots()

    ax.scatter(X[25:], y[25:], label="Normal Data")
    ax.scatter(X[:25], y[:25], label="Artificial Outliers")

    ax.set_title("Dataset with Artificial Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


def fit_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return float(model.coef_[0])


def fit_huber_regression(X, y):
    model = HuberRegressor()
    model.fit(X, y)
    return float(model.coef_[0])


def fit_ransac_regression(X, y):
    model = RANSACRegressor(random_state=42)
    model.fit(X, y)

    return float(model.estimator_.coef_[0])


def fit_theilsen_regression(X, y):
    model = TheilSenRegressor(random_state=42)
    model.fit(X, y)

    return float(model.coef_[0])


def coefficient_errors(coefs, true_coefficient):
    errors = {}

    for name, coef in coefs.items():
        errors[name] = abs(coef - true_coefficient)

    return errors


def best_robust_model(errors):
    robust_errors = {
        "huber_regression": errors["huber_regression"],
        "ransac_regression": errors["ransac_regression"],
        "theilsen_regression": errors["theilsen_regression"],
    }

    return min(robust_errors, key=robust_errors.get)


def ransac_outlier_summary(X, y):
    model = RANSACRegressor(random_state=42)
    model.fit(X, y)

    inliers = model.inlier_mask_
    outliers = ~inliers

    total_outliers_detected = int(np.sum(outliers))
    added_outliers_detected = int(np.sum(outliers[:25]))

    return total_outliers_detected, added_outliers_detected


def plot_regression_fits(X, y):
    fig, ax = plt.subplots()

    ax.scatter(X, y, alpha=0.6)

    x_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

    lr = LinearRegression().fit(X, y)
    huber = HuberRegressor().fit(X, y)
    ransac = RANSACRegressor(random_state=42).fit(X, y)
    theilsen = TheilSenRegressor(random_state=42).fit(X, y)

    ax.plot(x_line, lr.predict(x_line), label="Linear Regression")
    ax.plot(x_line, huber.predict(x_line), label="Huber")
    ax.plot(x_line, ransac.predict(x_line), label="RANSAC")
    ax.plot(x_line, theilsen.predict(x_line), label="Theil-Sen")

    ax.set_title("Regression Model Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


def plot_ransac_inliers_outliers(X, y):
    model = RANSACRegressor(random_state=42)
    model.fit(X, y)

    inliers = model.inlier_mask_
    outliers = ~inliers

    fig, ax = plt.subplots()

    ax.scatter(X[inliers], y[inliers], label="Inliers")
    ax.scatter(X[outliers], y[outliers], label="Outliers")

    ax.set_title("RANSAC Inliers and Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig
