from capymoa.stream import Stream
import pandas as pd
import json
import csv
import os
from datetime import datetime


class PrequentialResults:
    def __init__(
        self,
        learner: str = None,
        stream: Stream = None,
        wallclock: float = None,
        cpu_time: float = None,
        max_instances: int = None,
        cumulative_evaluator=None,
        windowed_evaluator=None,
        ground_truth_y=None,
        predictions=None,
        other_metrics=None,
    ):
        # protected attributes accessible through methods
        self._wallclock = wallclock
        self._cpu_time = cpu_time
        self._max_instances = max_instances
        self._ground_truth_y = ground_truth_y
        self._predictions = predictions
        self._other_metrics = other_metrics
        # attributes
        #: The name of the learner
        self.learner: str = learner
        #: The stream used to evaluate the learner
        self.stream: Stream = stream
        #: The cumulative evaluator
        self.cumulative = cumulative_evaluator
        #: The windowed evaluator
        self.windowed = windowed_evaluator

    def __getitem__(self, key):
        # Check if the key is a method of this class or an attribute
        if hasattr(self, key):
            if callable(getattr(self, key)):
                # If it is a method, then invoke it and return its return value
                return getattr(self, key)()
            else:
                return getattr(self, key)

        if hasattr(self.cumulative, key) and callable(getattr(self.cumulative, key)):
            return getattr(self.cumulative, key)()

        raise KeyError(f"Key {key} not found")

    def __getattr__(self, attribute):
        # Check if the attribute exists in the cumulative object
        if hasattr(self.cumulative, attribute):
            return getattr(self.cumulative, attribute)
        else:
            raise AttributeError(f"Attribute {attribute} not found")

    def write_to_file(self, path: str = "./", directory_name: str = None):
        if directory_name is None:
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory_name = f"{current_datetime}_{self._learner}"

        _write_results_to_files(path=path, results=self, directory_name=directory_name)

    def wallclock(self):
        return self._wallclock

    def cpu_time(self):
        return self._cpu_time

    def max_instances(self):
        return self._max_instances

    def ground_truth_y(self):
        return self._ground_truth_y

    def predictions(self):
        return self._predictions

    def other_metrics(self):
        return self._other_metrics

    def metrics_per_window(self):
        return self.windowed.metrics_per_window()


def _write_results_to_files(path: str = None, results=None, directory_name: str = None):
    from capymoa.evaluation import (
        ClassificationWindowedEvaluator,
        RegressionWindowedEvaluator,
        ClassificationEvaluator,
        RegressionEvaluator,
    )

    if results is None:
        raise ValueError("The results object is None")

    path = path if path.endswith("/") else (path + "/")

    if isinstance(results, ClassificationWindowedEvaluator) or isinstance(
        results, RegressionWindowedEvaluator
    ):
        data = results.metrics_per_window()
        data.to_csv(("./" if path is None else path) + "/windowed.csv", index=False)
    elif isinstance(results, ClassificationEvaluator) or isinstance(
        results, RegressionEvaluator
    ):
        json_str = json.dumps(results.metrics_dict())
        data = json.loads(json_str)
        with open(
            ("./" if path is None else path) + "/cumulative.csv", "w", newline=""
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data.keys())
            writer.writerow(data.values())
    elif isinstance(results, PrequentialResults):
        directory_name = (
            "prequential_results" if directory_name is None else directory_name
        )
        if os.path.exists(path + "/" + directory_name):
            raise ValueError(
                f"Directory {directory_name} already exists, please use another name"
            )
        else:
            os.makedirs(path + "/" + directory_name)

        _write_results_to_files(
            path=path + "/" + directory_name, results=results.cumulative
        )
        _write_results_to_files(
            path=path + "/" + directory_name, results=results.windowed
        )

        # If the ground truth and predictions are available, they will be writen to a file
        if results.ground_truth_y() is not None and results.predictions() is not None:
            y_vs_predictions = {
                "ground_truth_y": results.ground_truth_y(),
                "predictions": results.predictions(),
            }
            if len(y_vs_predictions) > 0:
                t_p = pd.DataFrame(y_vs_predictions)
                t_p.to_csv(
                    ("./" if path is None else path)
                    + "/"
                    + directory_name
                    + "/ground_truth_y_and_predictions.csv",
                    index=False,
                )
    else:
        raise ValueError(
            "Writing results to file is not supported for type "
            + str(type(results))
            + " yet"
        )
