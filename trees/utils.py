import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(
    model, X: np.ndarray, y: np.ndarray, pts_per_interval: int = 10
):
    assert pts_per_interval > 0
    assert hasattr(model, "predict"), "Model must have predict method"

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    delta_x = (x_max - x_min) / pts_per_interval
    x_min -= delta_x
    x_max += delta_x

    delta_y = (y_max - y_min) / pts_per_interval
    y_min -= delta_y
    y_max += delta_y

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, delta_x), np.arange(y_min, y_max, delta_y)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.5, cmap=plt.cm.Paired)
    plt.scatter(*X.T, c=y, cmap=plt.cm.Paired, edgecolors="black")
    plt.show()
