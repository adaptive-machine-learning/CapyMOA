from abc import ABC, abstractmethod

from capymoa.stream import Stream
from capymoa._utils import _translate_metric_name
import pandas as pd
import json
import csv
import os
from datetime import datetime


class Results(ABC):
    """
    Abstract base class for Evaluation Results.
    Attributes:
    - dict_results: The dictionary with the results from an evaluation. These results are produced by high-level
    evaluation functions such as prequential_evaluation(...)
    - schema: The schema representing the instances. Defaults to None.


    See also prequential_evaluation,
    """

    def __init__(
            self,
            dict_results: dict,
    ):
        self.dict_results = dict_results

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def learner(self):
        pass

    @abstractmethod
    def stream(self):
        pass

    @abstractmethod
    def wallclock(self):
        pass

    @abstractmethod
    def cpu_time(self):
        pass

    @abstractmethod
    def max_instances(self):
        pass


class PrequentialResults(Results):
    def __init__(self,
                 learner: str = None,
                 stream: Stream = None,
                 wallclock: float = None,
                 cpu_time: float = None,
                 max_instances: int = None,
                 cumulative_evaluator=None,
                 windowed_evaluator=None,
                 ground_truth_y=None,
                 predictions=None,
                 other_metrics=None):

        self._learner = learner
        self._stream = stream
        self._wallclock = wallclock
        self._cpu_time = cpu_time
        self._max_instances = max_instances
        self._ground_truth_y = ground_truth_y
        self._predictions = predictions
        self._other_metrics = other_metrics
        self.cumulative = cumulative_evaluator
        self.windowed = windowed_evaluator

        self.dict_results = {
            "learner": learner,
            "stream": stream,
            "wallclock": wallclock,
            "cpu_time": cpu_time,
            "max_instances": max_instances,
            "cumulative": cumulative_evaluator,
            "windowed": windowed_evaluator,
            "ground_truth_y": ground_truth_y,
            "predictions": predictions,
            "other_metrics": other_metrics,
            }

        super().__init__(self.dict_results)

    def __getitem__(self, key):
        value = self.dict_results.get(key, None)
        # In case the key is a metric, then attempt to access the cumulative metrics through __getattr__
        if value is None:
            return self.__getattr__(key)()
        return value

    def __getattr__(self, attribute):
        cumulative_obj = self.dict_results['cumulative']
        if hasattr(cumulative_obj, attribute):
            attr = getattr(cumulative_obj, attribute)
            return attr
        return None

    def __str__(self):
        return str(self.dict_results)

    def write_to_file(self, path: str = './', directory_name: str = None):
        if directory_name is None:
            current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
            directory_name = f"{current_datetime}_{self._learner}"

        _write_results_to_files(
            path=path,
            results=self,
            directory_name=directory_name
        )

    def learner(self):
        return self.learner

    def stream(self):
        return self._stream

    def wallclock(self):
        return self._wallclock

    def cpu_time(self):
        return self._cpu_time

    def max_instances(self):
        return self._max_instances

    def ground_truth_y(self):
        return self.dict_results.get('ground_truth_y')

    def predictions(self):
        return self.dict_results.get('predictions')

    def other_metrics(self):
        return self._other_metrics

    def metrics_per_window(self):
        return self.windowed.metrics_per_window()


class AnomalyDetectionResults:
    def __init__(
            self,
            cumulative_evaluator=None,
            windowed_evaluator=None
    ):
        self.cumulative_evaluator = cumulative_evaluator
        self.windowed_evaluator = windowed_evaluator

    def auc(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_translate_metric_name('auc', to='moa')]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_translate_metric_name('auc', to='moa')].tolist()
        else:
            raise ValueError('no results found')

    def s_auc(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_translate_metric_name('s_auc', to='moa')]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_translate_metric_name('s_auc', to='moa')].tolist()
        else:
            raise ValueError('no results found')


def _write_results_to_files(
        path: str = None,
        results=None,
        directory_name: str = None
):
    from capymoa.evaluation import (ClassificationWindowedEvaluator,
                                    RegressionWindowedEvaluator,
                                    ClassificationEvaluator,
                                    RegressionEvaluator)

    if results is None:
        raise ValueError('The results object is None')

    path = path if path.endswith('/') else (path + '/')

    if isinstance(results, ClassificationWindowedEvaluator) or isinstance(results, RegressionWindowedEvaluator):
        data = results.metrics_per_window()
        data.to_csv(('./' if path is None else path) + f'/windowed.csv', index=False)
    elif isinstance(results, ClassificationEvaluator) or isinstance(results, RegressionEvaluator):
        json_str = json.dumps(results.metrics_dict())
        data = json.loads(json_str)
        with open(('./' if path is None else path) + f"/cumulative.csv", 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data.keys())
            writer.writerow(data.values())
    elif isinstance(results, PrequentialResults):
        directory_name = 'prequential_results' if directory_name is None else directory_name
        if os.path.exists(path + '/' + directory_name):
            raise ValueError(f'Directory {directory_name} already exists, please use another name')
        else:
            os.makedirs(path + '/' + directory_name)

        _write_results_to_files(path=path + '/' + directory_name, results=results.cumulative)
        _write_results_to_files(path=path + '/' + directory_name, results=results.windowed)

        # If the ground truth and predictions are available, they will be writen to a file
        if results.ground_truth_y() is not None and results.predictions() is not None:
            y_vs_predictions = {'ground_truth_y': results.ground_truth_y(),
                                'predictions': results.predictions()}
            if len(y_vs_predictions) > 0:
                t_p = pd.DataFrame(y_vs_predictions)
                t_p.to_csv(('./' if path is None else path) + '/' + directory_name +
                           '/ground_truth_y_and_predictions.csv',
                           index=False)
    else:
        raise ValueError('Writing results to file is not supported for type ' + str(type(results)) + ' yet')
