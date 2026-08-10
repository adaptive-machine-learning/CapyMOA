import numpy as np

from capymoa.base._base import AnomalyDetector
from capymoa.core import Instance
from capymoa.stream._stream import Schema


def optimal_histogram_birge_rozenholc(x, d_max=None):
    """Optimal regular (equi-width) histogram via the penalized maximum
    likelihood rule of Birge & Rozenholc (2006).

    Maximizes  sum_j N_j * log(D * N_j / n) - [(D - 1) + (log D)^2.5]
    over 1 <= D <= floor(n / log n).

    :param x: 1-D sample of real values.
    :param d_max: Largest number of bins to test. Defaults to ``floor(n / log n)``.
    :return: A tuple ``(best_d, counts, edges, scores)`` where ``best_d`` is the
        optimal number of bins, ``counts`` is the bin counts of length ``best_d``,
        ``edges`` is the bin edges of length ``best_d + 1``, and ``scores`` is the
        penalized log-likelihood for each candidate D (index 0 → D = 1).
    """

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size

    if n < 2 or x.min() == x.max():  # trivial / degenerate cases
        lo = float(x.min()) if n else 0.0
        hi = float(x.max()) if n else 1.0
        if hi == lo:
            hi = lo + 1.0
        return 1, np.array([n]), np.array([lo, hi]), np.array([0.0])

    xmin, xmax = x.min(), x.max()
    if d_max is None:
        d_max = max(1, int(np.floor(n / np.log(n))))

    scores = np.empty(d_max)
    best_d = 1
    best_score = -np.inf
    best_counts = None
    best_edges = None

    for D in range(1, d_max + 1):
        counts, edges = np.histogram(x, bins=D, range=(xmin, xmax))
        nz = counts[counts > 0]
        loglik = np.sum(nz * np.log(D * nz / n))
        penalty = (D - 1) + np.log(D) ** 2.5  # 0 when D == 1
        score = loglik - penalty
        scores[D - 1] = score

        if score > best_score:
            best_score = score
            best_d = D
            best_counts = counts
            best_edges = edges

    return best_d, best_counts, best_edges, scores


class Loda(AnomalyDetector):
    """Loda: Lightweight on-line detector of anomalies
    We implement a streaming version of Loda that updates the histograms after every window of instances.

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.anomaly import Loda
    >>> from capymoa.evaluation import AnomalyDetectionEvaluator
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = Loda(schema, n_projections=10, window_size=100, random_state=42)
    >>> evaluator = AnomalyDetectionEvaluator(schema)
    >>> while stream.has_more_instances():
    ...     instance = stream.next_instance()
    ...     proba = learner.score_instance(instance)
    ...     evaluator.update(instance.y_index, proba)
    ...     learner.train(instance)
    >>> auc = evaluator.auc()
    >>> print(f"AUC: {auc:.2f}")
    AUC: 0.65

    Reference:
        Pevný, T. (2016). Loda: Lightweight on-line detector of anomalies. Machine Learning, 102(2), 275-304.
    """

    def __init__(
        self,
        schema: Schema,
        n_projections: int = 100,
        window_size: int = 256,
        max_bins: str | int = "auto",
        random_state: int = 42,
    ):
        """Initialize the Loda anomaly detector.

        :param schema: Schema of the data stream.
        :param n_projections: Number of random projection histograms in the ensemble.
        :param window_size: Number of recent instances used to fit each histogram.
        :param max_bins: Upper bound on bins per histogram. ``"auto"`` uses ``floor(n / log n)``
            via the Birge-Rozenholc criterion; an integer caps the search at that value.
        :param random_state: Random seed for reproducibility.
        """

        super().__init__(schema=schema, random_seed=random_state)
        self.n_projections = n_projections
        self.window_size = window_size
        self.max_bins = max_bins
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.n = 0  # number of instances seen
        self._setup()

    def _setup(self):
        # initialize random projection matrix W
        d = self.schema.get_num_attributes()
        k = self.n_projections
        m = int(np.sqrt(d))
        self.W = self.rng.standard_normal((k, d))
        keep = np.argsort(self.rng.random((k, d)), axis=1)[
            :, :m
        ]  # m random cols per row
        mask = np.zeros((k, d), dtype=bool)
        np.put_along_axis(mask, keep, True, axis=1)
        self.W[~mask] = 0.0

        # initialize sliding window for each projection
        self.windows = np.zeros((k, self.window_size))

        # initialize histograms for each projection
        self.histograms = None

    def train(self, instance: Instance):
        """Train on a single instance.

        Projects the instance onto each random projection vector and adds it to
        the circular window. Histograms are rebuilt once every ``window_size``
        instances using the Birge-Rozenholc criterion to select the optimal
        number of bins.

        :param instance: The instance to train on.
        """
        # project instance onto random projections
        projections = self.W @ instance.x

        # update sliding windows
        self.windows[:, self.n % self.window_size] = projections
        self.n += 1

        if self.n % self.window_size != 0:
            return  # wait until we have a full window

        # update histograms
        if self.max_bins == "auto":
            n_bins = None
        else:
            n_bins = self.max_bins
        self.histograms = []
        for i in range(self.n_projections):
            _, hist, bin_edges, _ = optimal_histogram_birge_rozenholc(
                self.windows[i], d_max=n_bins
            )
            assert hist is not None and bin_edges is not None
            # Laplace smoothing avoids zero probabilities (and -inf log-likelihoods)
            # for bins that happened to receive no samples in this window.
            probs = (hist + 1) / (hist.sum() + len(hist))
            self.histograms.append((probs, bin_edges))

    def score_instance(self, instance: Instance) -> float:
        """Return the anomaly score for a single instance.

        Computes the average negative log-likelihood of the instance under
        the ensemble of one-dimensional histograms. Higher scores indicate
        greater anomalousness. Returns ``0.0`` before the first full window
        of ``window_size`` instances has been seen.

        :param instance: The instance to score.
        :return: Anomaly score. Ranges from 0.0 (least anomalous) to infinity (most anomalous).
        """
        if self.histograms is None:
            return 0.0

        projections = self.W @ instance.x
        log_probs = []
        for i in range(self.n_projections):
            probs, bin_edges = self.histograms[i]
            bin_index = np.searchsorted(bin_edges, projections[i], side="right") - 1
            bin_index = np.clip(bin_index, 0, len(probs) - 1)
            log_probs.append(np.log(probs[bin_index]))
        return float(-np.mean(log_probs))

    def predict(self, instance: Instance) -> int:
        raise NotImplementedError(
            "Loda does not implement predict. Use score_instance instead."
        )
