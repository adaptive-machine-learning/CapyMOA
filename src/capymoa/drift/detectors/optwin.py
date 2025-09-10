from capymoa.drift.base_detector import BaseDriftDetector
import numpy as np
from scipy.stats import t as t_stat
from scipy.optimize import fsolve
import scipy.stats
import math
import warnings
from typing import Any, Dict


class OPTWIN(BaseDriftDetector):
    """Optimal Window Concept Drift Detector

    Drift Identification with Optimal Sub-Windows (OPTWIN) [#0]_ is a drift detection
    method.

    >>> import numpy as np
    >>> from capymoa.drift.detectors import OPTWIN
    >>> np.random.seed(0)
    >>>
    >>> detector = OPTWIN(rigor=0.1, drift_confidence=0.9)
    >>>
    >>> data_stream = np.random.randint(2, size=2000)
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>>
    >>> for i in range(2000):
    ...     detector.add_element(data_stream[i])
    ...     if detector.detected_change():
    ...         print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
    Change detected in data: 6 - at index: 1164


    .. [#0] Tosi, Mauro D. L., and Martin Theobald. “OPTWIN: Drift Identification with
        Optimal Sub-Windows.” 2024 IEEE 40th International Conference on Data
        Engineering Workshops (ICDEW), 2024.
    """

    class _Circular_list:
        """Support class for OPTWIN, implements a circular list with fixed maximum size."""

        def __init__(self, maxSize: int):
            self.W = [0 for i in range(maxSize)]
            self.length = 0
            self.init = 0

        def pos(self, idx: int) -> int:
            """Get the real position in the circular list."""
            if self.init + idx < len(self.W):
                return self.init + idx
            else:
                return self.init + idx - len(self.W)

        def add(self, element: float) -> None:
            """Add an element to the end of the circular list."""
            position = self.pos(self.length)
            self.length = self.length + 1
            self.W[position] = element

        def pop_first(self) -> float:
            """Pop the first element of the circular list."""
            element = self.W[self.init]
            position = self.pos(1)
            self.init = position
            self.length = self.length - 1
            return element

        def get(self, idx: int) -> float:
            """Get the element at index idx."""
            position = self.pos(idx)
            return self.W[position]

        def get_interval(self, idx1: int, idx2: int) -> list[float]:
            """Get the elements between idx1 and idx2 (not including idx2)."""
            position1 = self.pos(idx1)
            position2 = self.pos(idx2)
            if position1 <= position2:
                return self.W[position1:position2]
            else:
                return self.W[position1:] + self.W[:position2]

    def __init__(
        self,
        rigor: float = 0.5,
        drift_confidence: float = 0.999,
        warning_confidence: float = 0.9,
        empty_w: bool = True,
        w_length_max: int = 1_000,
        w_length_min: int = 30,
        minimum_noise: float = 1e-6,
    ):
        """Initialize the OPTWIN drift detector.

        :param rigor: Rigorousness of drift identification
        :param drift_confidence: Confidence value chosen by user
        :param warning_confidence: Confidence value for warning zone
        :param empty_w: Empty window when drift is detected
        :param w_length_max: Maximum window size. 25000 is recommended but slows down
            initialization as it pre-computes optimal cuts for all window sizes up to
            ``w_length_max``.
        :param w_length_min: Minimum window size
        :param minimum_noise: Noise to be added to stdev in case it is 0
        """
        super().__init__()
        warnings.filterwarnings(
            "ignore", message="The iteration is not making good progress"
        )
        warnings.filterwarnings(
            "ignore", message="divide by zero encountered in divide"
        )
        warnings.filterwarnings(
            "ignore", message="divide by zero encountered in scalar divide"
        )
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")

        # OPTWIN parameters
        self.drift_confidence = drift_confidence
        self.rigor = rigor
        self.w_length_max = w_length_max
        self.w_length_min = w_length_min
        self.minimum_noise = minimum_noise
        self.pre_compute_optimal_cut = True
        self.empty_w = empty_w

        # Data storage
        self.W = self._Circular_list(w_length_max)
        self.opt_cut = []
        self.opt_phi = []
        self.t_stats = []
        self.t_stats_warning = []
        self.last_opt_cut = 0
        self.drift_type = []
        self.confidence = pow(self.drift_confidence, 1 / 4)
        self.confidence_warning = warning_confidence

        self.t_score = lambda n: t_stat.ppf(self.confidence, df=self.degree_freedom(n))
        self.t_score_warning = lambda n: t_stat.ppf(
            self.confidence_warning, df=self.degree_freedom(n)
        )
        self.f_test = lambda n: scipy.stats.f.ppf(
            q=self.confidence,
            dfn=(n * self.W.length) - 1,
            dfd=self.W.length - (n * self.W.length) - 1,
        )
        self.degree_freedom = lambda n: pow(
            (
                (1 / max(self.W.length * n, 1e-15))
                + ((1 / pow(self.f_test(n), 2)) / ((1 - n) * self.W.length))
            ),
            2,
        ) / (
            (1 / max((pow((self.W.length * n), 2) * ((self.W.length * n) - 1)), 1e-15))
            + (
                pow((1 / pow(self.f_test(n), 2)), 2)
                / max(
                    (
                        pow(((1 - n) * self.W.length), 2)
                        * (((1 - n) * self.W.length) - 1)
                    ),
                    1e-15,
                )
            )
        )
        self.t_test = lambda n: self.rigor - (
            self.t_score(n)
            * np.sqrt(
                (1 / (self.W.length * n))
                + ((1 * self.f_test(n)) / ((1 - n) * self.W.length))
            )
        )

        # Running stdev and avg
        self.stdev_new = 0
        self.summation_new = 0
        self.count_new = 0
        self.S_new = 0
        self.stdev_h = 0
        self.summation_h = 0
        self.count_h = 0
        self.S_h = 0

        self.in_concept_change = False
        self.in_warning_zone = False

        # Pre-compute optimal cut for all possible window sizes
        if self.pre_compute_optimal_cut:
            self._pre_compute_cuts()

        if len(self.opt_cut) == 0:
            self.opt_cut = [0 for i in range(w_length_min)]
            self.opt_phi = [0 for i in range(w_length_min)]
            self.t_stats = [0.0 for i in range(w_length_min)]
            self.t_stats_warning = [0.0 for i in range(w_length_min)]

        if len(self.opt_cut) >= w_length_max and len(self.opt_phi) >= w_length_max:
            self.pre_compute_optimal_cut = True

    # TODO: Add caching of cuts for standard parameters
    def _pre_compute_cuts(self) -> None:
        """Pre-compute optimal cuts and phi values for all possible window sizes."""

        if (
            len(self.opt_cut) != 0
            and len(self.opt_phi) != 0
            and len(self.t_stats) != 0
            and len(self.t_stats_warning) != 0
        ):
            return  # Already computed

        self.W = self._Circular_list(self.w_length_max)
        for i in range(self.w_length_max + 1):
            if i < self.w_length_min:
                self.opt_cut.append(0)
                self.opt_phi.append(0)
                self.t_stats.append(0.0)
                self.t_stats_warning.append(0.0)
            else:
                optimal_cut = fsolve(self.t_test, (self.W.length - 30) / self.W.length)

                tolerance = 1e-6
                if abs(self.t_test(optimal_cut[0])) <= tolerance:
                    optimal_cut = math.floor(optimal_cut[0] * self.W.length)
                else:
                    optimal_cut = math.floor((self.W.length / 2) + 1)

                phi_opt = scipy.stats.f.ppf(
                    q=self.confidence,
                    dfn=optimal_cut - 1,
                    dfd=self.W.length - optimal_cut - 1,
                )
                self.opt_cut.append(optimal_cut)
                self.opt_phi.append(phi_opt)
                self.t_stats.append(self.t_score(optimal_cut / i))
                self.t_stats_warning.append(self.t_score_warning(optimal_cut / i))
            # Dummy add to window to increase its size and continue pre-computation
            self.W.add(1)

        self.W = self._Circular_list(self.w_length_max)

    def _insert_to_W(self, element: float) -> None:
        self.W.add(element)
        self._add_running_stdev("new", [element])

        # If window is too large, remove the oldest element
        if self.W.length > self.w_length_max:
            pop = self.W.pop_first()
            self._pop_from_running_stdev("h", [pop])
            self._pop_from_running_stdev("new", [self.W.get(self.last_opt_cut)])
            self._add_running_stdev("h", [self.W.get(self.last_opt_cut)])
        return

    def _add_running_stdev(self, window: str, element: list[float]) -> None:
        """Update running stdev and avg by adding elements element.

        :param window: "new" or "h", indicating which window to update
        :param element: List of elements to add"""

        if window == "new":
            summation = self.summation_new
            count = self.count_new
            S = self.S_new
        else:
            summation = self.summation_h
            count = self.count_h
            S = self.S_h

        summation += sum(element)
        count += len(element)
        S += sum([i * i for i in element])

        if count > 1 and S > 0:
            stdev = math.sqrt((count * S) - (summation * summation)) / count
        else:
            stdev = 0

        if window == "new":
            self.summation_new = summation
            self.count_new = count
            self.S_new = S
            self.stdev_new = stdev
        else:
            self.summation_h = summation
            self.count_h = count
            self.S_h = S
            self.stdev_h = stdev

    def _pop_from_running_stdev(self, window: str, element: list[float]) -> None:
        """Update running stdev and avg by removing elements element.

        :param window: "new" or "h", indicating which window to update
        :param element: List of elements to remove"""

        if window == "new":
            summation = self.summation_new
            count = self.count_new
            S = self.S_new
        else:
            summation = self.summation_h
            count = self.count_h
            S = self.S_h
        summation -= sum(element)
        count -= len(element)
        S -= sum([i * i for i in element])

        if count > 1 and S > 0:
            stdev = math.sqrt((count * S) - (summation * summation)) / count
        else:
            stdev = 0

        if window == "new":
            self.summation_new = summation
            self.count_new = count
            self.S_new = S
            self.stdev_new = stdev
        else:
            self.summation_h = summation
            self.count_h = count
            self.S_h = S
            self.stdev_h = stdev

    def add_element(self, element: float) -> None:
        """
        Add the new element and perform change detection

        :param element: The new observation
        """
        self.idx += 1
        self.data.append(element)
        self._insert_to_W(element)

        # Check if window is too small
        if self.W.length < self.w_length_min:
            self.in_concept_change = False
            self.in_warning_zone = False
            return

        # check optimal window cut and phi
        # get pre-calculated optimal window cut and phi
        optimal_cut = self.opt_cut[self.W.length]
        phi_opt = self.opt_phi[self.W.length]

        # Update running stdev and avg
        if (
            optimal_cut > self.last_opt_cut
        ):  # Remove elements from window_new and add them to window_h
            self._pop_from_running_stdev(
                "new", self.W.get_interval(self.last_opt_cut, optimal_cut)
            )
            self._add_running_stdev(
                "h", self.W.get_interval(self.last_opt_cut, optimal_cut)
            )

        elif (
            optimal_cut < self.last_opt_cut
        ):  # Remove elements from window_h and add them to window_new
            self._pop_from_running_stdev(
                "h", self.W.get_interval(optimal_cut, self.last_opt_cut)
            )
            self._add_running_stdev(
                "new", self.W.get_interval(optimal_cut, self.last_opt_cut)
            )

        avg_h = self.summation_h / self.count_h
        avg_new = self.summation_new / self.count_new

        stdev_h = (
            math.sqrt((self.count_h * self.S_h) - (self.summation_h * self.summation_h))
            / self.count_h
        )
        stdev_new = (
            math.sqrt(
                (self.count_new * self.S_new)
                - (self.summation_new * self.summation_new)
            )
            / self.count_new
        )

        self.last_opt_cut = optimal_cut

        # Add minimal noise to stdev
        stdev_h += self.minimum_noise
        stdev_new += self.minimum_noise

        if self.pre_compute_optimal_cut:
            t_stat = self.t_stats[self.W.length]
            t_stat_warning = self.t_stats_warning[self.W.length]
        else:
            t_stat = self.t_score(optimal_cut / self.W.length)
            t_stat_warning = self.t_score_warning(optimal_cut / self.W.length)

        # Actual drift detection

        # t-test
        t_test_result = (avg_new - avg_h) / (
            math.sqrt(
                (stdev_new / (self.W.length - optimal_cut)) + (stdev_h / optimal_cut)
            )
        )
        if t_test_result > t_stat:
            self._drift_reaction("t")
            return
        elif t_test_result > t_stat_warning:
            self._warning_reaction("t")
            # dont return as f-test might find a drift
        else:
            self.in_warning_zone = False
            self.in_concept_change = False

        # f-test
        if (stdev_new * stdev_new / (stdev_h * stdev_h)) > phi_opt:
            if avg_h - avg_new < 0:  # Performance is degrading, detecting drift
                self._drift_reaction("f")
                return
            else:  # Performance is improving, no drift detected but we still empty the window
                self._empty_window()
                self.in_concept_change = False
                self.in_warning_zone = False

    def _empty_window(self) -> None:
        """Empty the window, resetting all its parameters."""
        self.W = self._Circular_list(self.w_length_max)
        self.stdev_new = 0
        self.summation_new = 0
        self.count_new = 0
        self.S_new = 0
        self.stdev_h = 0
        self.summation_h = 0
        self.count_h = 0
        self.S_h = 0
        self.last_opt_cut = 0

    def _drift_reaction(self, drift_type: str) -> None:
        """Reaction to a detected drift.

        :param drift_type: Type of drift detected ("t" or "f")
        """
        self.detection_index.append(self.idx)
        self.drift_type.append(drift_type)
        self.in_concept_change = True
        self.in_warning_zone = False
        self._empty_window()

    def _warning_reaction(self, drift_type: str) -> None:
        """Reaction to a detected warning.

        :param drift_type: Type of warning detected ("t" or "f")
        """
        self.warning_index.append(self.idx)
        self.drift_type.append(drift_type)
        self.in_warning_zone = True
        self.in_concept_change = False

    def get_params(self) -> Dict[str, Any]:
        """Get the hyper-parameters of the OPTWIN drift detector."""
        return {
            "rigor": self.rigor,
            "drift_confidence": self.drift_confidence,
            "warning_confidence": self.confidence_warning,
            "w_length_max": self.w_length_max,
            "w_length_min": self.w_length_min,
        }
