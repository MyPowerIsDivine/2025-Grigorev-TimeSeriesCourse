import numpy as np
from collections import Counter
from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize

default_metrics_params = {'euclidean': {'normalize': True},
                         'dtw': {'normalize': True, 'r': 0.05}
                         }

class TimeSeriesKNN:
    """
    KNN Time Series Classifier

    Parameters
    ----------
    n_neighbors: number of neighbors
    metric: distance measure between time series
             Options: {euclidean, dtw}
    metric_params: dictionary containing parameters for the distance metric being used
    """
    
    def __init__(self, n_neighbors: int = 3, metric: str = 'euclidean', metric_params: dict | None = None) -> None:

        self.n_neighbors: int = n_neighbors
        self.metric: str = metric
        self.metric_params: dict | None = default_metrics_params[metric].copy()
        if metric_params is not None:
            self.metric_params.update(metric_params)


    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        Fit the model using X_train as training data and Y_train as labels
        """
        self.X_train = X_train
        self.Y_train = Y_train

        return self


    def _distance(self, x_train: np.ndarray, x_test: np.ndarray) -> float:
        """
        Compute distance between the train and test samples
        """
        dist = 0
        normalize = self.metric_params.get('normalize', False)

        if self.metric == 'euclidean':
            if normalize:
                dist = norm_ED_distance(x_train, x_test)
            else:
                dist = ED_distance(x_train, x_test)
        
        elif self.metric == 'dtw':
            r = self.metric_params.get('r', 1)
            if normalize:
                dist = DTW_distance(z_normalize(x_train), z_normalize(x_test), r)
            else:
                dist = DTW_distance(x_train, x_test, r)
        else:
             raise ValueError(f"Unknown metric: {self.metric}")

        return dist


    def _find_neighbors(self, x_test: np.ndarray) -> list[tuple[float, int]]:
        """
        Find the k nearest neighbors of the test sample
        Returns: list of (distance, label)
        """
        distances = []

        for i in range(len(self.X_train)):
            d = self._distance(self.X_train[i], x_test)
            label = self.Y_train[i]
            distances.append((d, label))

        distances.sort(key=lambda x: x[0])

        neighbors = distances[:self.n_neighbors]

        return neighbors


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for samples of the test set
        """
        y_pred = []

        for x in X_test:
            neighbors = self._find_neighbors(x)
            
            neighbor_labels = [label for (_, label) in neighbors]
            
            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            y_pred.append(most_common)

        return np.array(y_pred)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy classification score
    """
    score = 0
    for i in range(len(y_true)):
        if (y_pred[i] == y_true[i]):
            score = score + 1
    score = score/len(y_true)

    return score