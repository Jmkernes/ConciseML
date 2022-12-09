from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles, make_moons, make_blobs

from trees.utils import plot_decision_boundary
from trees.models import RandomForest


def main(max_depth: int = 5, n_trees: int = 20):
    model = RandomForest()
    names = ["make_moons", "make_circles", "make_blobs"]
    for i, (X, y) in enumerate(
        [
            make_moons(noise=0.25),
            make_circles(noise=0.15, factor=0.5),
            make_blobs(centers=5),
        ]
    ):
        y_pred = model.fit(X, y, max_depth=max_depth, n_trees=n_trees).predict(X)
        print(
            f"Dataset: {names[i]}. Training accuracy: {100 * accuracy_score(y, y_pred):.02f}"
        )
        plot_decision_boundary(model, X, y)


if __name__ == "__main__":
    main()
