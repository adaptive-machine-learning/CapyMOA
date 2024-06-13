from abc import ABC, abstractmethod
import json, csv, os
from capymoa.stream import Stream, Schema
import pandas as pd
import numpy as np
import time
import warnings


def write_files(
        path: str | None = None,
        results=None,
        file_name: str | None = None,
        directory_name: str | None = None,
):

    if results is None:
        raise ValueError('nothing to write')

    path = path if path.endswith('/') else (path+'/')

    if isinstance(results, CumulativeResults):
        json_str = json.dumps(results.cumulative.metrics_dict())
        data = json.loads(json_str)
        with open(('./' if path is None else path)+('/cumulative_results.csv' if file_name is None else file_name), 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data.keys())
            writer.writerow(data.values())
    elif isinstance(results, WindowedResults):
        data = results.metrics_per_window()
        data.to_csv(('./' if path is None else path)+('/windowed_results.csv' if file_name is None else file_name), index=False)
    elif isinstance(results, dict):
        json_str = json.dumps(results)
        data = json.loads(json_str)
        with open(('./' if path is None else path)+('/results.csv' if file_name is None else file_name), 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data.keys())
            writer.writerow(data.values())
    elif isinstance(results, PrequentialResults):
        directory_name = 'prequential_results' if directory_name is None else directory_name
        if os.path.exists(path+'/'+directory_name):
            raise ValueError(f'directory {directory_name} exists, please use another name')
        else:
            os.makedirs(path+'/'+directory_name)

        write_files(path=path+'/'+directory_name, file_name=file_name, results=results.cumulative)
        write_files(path=path+'/'+directory_name, file_name=file_name, results=results.windowed)

        if results.prevent_saving_t_v_p is False:
            t_vs_p = {}
            if results.get_targets() is not None:
                t_vs_p['targets'] = results.get_targets()
            if results.get_predictions() is not None:
                t_vs_p['predictions'] = results.get_predictions()
            if len(t_vs_p) > 0:
                t_p = pd.DataFrame(t_vs_p)
                t_p.to_csv(('./' if path is None else path)+'/'+directory_name+ '/targets_vs_predictions.csv', index=False)
    else:
        raise ValueError('writing results to file is not supported for Type ' +str(type(results)) + ' yet')


class Results(ABC):
    """
    Abstract base class for Evaluation Results.
    # TODO: unfinished part
    Attributes:
    - schema: The schema representing the instances. Defaults to None.
    -
    """

    def __init__(
            self,
            dict_results: dict,
            schema: Schema,
    ):
        if self.schema is None:
            raise ValueError("Schema must be initialised")
        self.dict_results = dict_results
        self.schema = schema

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def to_file(self):
        pass

    @abstractmethod
    def learner(self):
        pass

    @abstractmethod
    def stream(self):
        pass

    @abstractmethod
    def schema(self):
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

class CumulativeResults(Results):
    def __init__(
            self,
            dict_results: dict = None,
            schema: Schema = None,
            stream: Stream = None,
            cumulative_evaluator=None,
            wallclock: float = None,
            cpu_time: float = None,
            max_instances: int = None,

    ):
        if dict_results is not None:
            self.schema = dict_results["stream"].get_schema()
            self.stream = dict_results["stream"]
            self.cumulative_evaluator = dict_results["cumulative"]
            self.wallclock = dict_results["wallclock"]
            self.cpu_time = dict_results["cpu_time"]
            self.max_instances = dict_results["max_instances"]
            self.dict_results = dict_results

        else:
            if schema is None:
                raise ValueError("Schema must be initialised")
            self.schema = schema
            self.stream = stream
            self.cumulative_evaluator = cumulative_evaluator
            self.wallclock = wallclock
            self.cpu_time = cpu_time
            self.max_instances = max_instances

            self.dict_results = {
                "stream": self.stream,
                "schema": self.schema,
                "cumulative": self.cumulative_evaluator.metrics_dict()['cumulative'],
                "wallclock": self.wallclock,
                "cpu_time": self.cpu_time,
                "max_instances": self.max_instances,
            }

    def __getitem__(self, key):
        return self.dict_results.get(key, None)

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, path=None):
        write_files(path=path, results=self)

    def learner(self):
        return self.learner

    def stream(self):
        return self.stream

    def schema(self):
        return self.stream.get_schema()

    def wallclock(self):
        return self.wallclock

    def cpu_time(self):
        return self.cpu_time

    def max_instances(self):
        return self.max_instances

    def full_targets(self):
        return self.full_targets

    def metrics_dict(self):
        return self.cumulative.metrics_dict()

    @property
    def cumulative(self):
        return self.cumulative_evaluator

    @property
    def results(self):
        return self.cumulative

    # Classification Metrics
    def accuracy(self):
        if 'classifications correct (percent)' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['classifications correct (percent)']
        else:
            raise ValueError("accuracy not exit")

    def kappa(self):
        if 'Kappa Statistic (percent)' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['Kappa Statistic (percent)']
        else:
            raise ValueError("Kappa statistic not exit")

    def kappa_T(self):
        if 'Kappa Temporal Statistic (percent)' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['Kappa Temporal Statistic (percent)']
        else:
            raise ValueError("Kappa temporal statistic not exit")

    def kappa_M(self):
        if 'Kappa M Statistic (percent)' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['Kappa M Statistic (percent)']
        else:
            raise ValueError("Kappa M-statistic not exit")

    def f1_score(self):
        if 'F1 Score (percent)' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['F1 Score (percent)']
        else:
            raise ValueError("F1 score not exit")

    def precision(self):
        if 'Precision (percent)' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['Precision (percent)']
        else:
            raise ValueError("Precision not exit")

    def recall(self):
        if 'Recall (percent)' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['Recall (percent)']
        else:
            raise ValueError("Recall not exit")

    # Regression Metrics
    def MAE(self):
        if 'mean absolute error' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['mean absolute error']
        else:
            raise ValueError("MAE not exit")

    def RMSE(self):
        if 'root mean squared error' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['root mean squared error']
        else:
            raise ValueError("RMSE not exit")

    def RMAE(self):
        if 'relative mean absolute error' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['relative mean absolute error']
        else:
            raise ValueError("RMAE not exit")

    def RRMSE(self):
        if 'relative root mean squared error' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['relative root mean squared error']
        else:
            raise ValueError("RRMSE not exit")

    def R2(self):
        if 'coefficient of determination' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['coefficient of determination']
        else:
            raise ValueError("R2 not exit")

    def adjusted_R2(self):
        if 'adjusted coefficient of determination' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['adjusted coefficient of determination']
        else:
            raise ValueError("adjusted R2 not exit")

    # Prediction Interval Metrics
    def coverage(self):
        if 'coverage' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['coverage']
        else:
            raise ValueError("Coverage not exit")

    def average_length(self):
        if 'average length' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['average length']
        else:
            raise ValueError("Average length not exit")

    def NMPIW(self):
        if 'NMPIW' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['NMPIW']
        else:
            raise ValueError("NMPIW not exit")

    # Anomaly Metrics
    def auc(self):
        if 'AUC' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['auc']
        else:
            raise ValueError("AUC not exit")

    def s_auc(self):
        if 'sAUC' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['sAUC']
        else:
            raise ValueError("sAUC not exit")


class WindowedResults(Results):
    def __init__(
            self,
            dict_results: dict = None,
            schema: Schema = None,
            stream: Stream = None,
            windowed_evaluator=None,
            wallclock: float = None,
            cpu_time: float = None,
            max_instances: int = None,
    ):
        if dict_results is not None:
            self.schema = dict_results["stream"].get_schema()
            self.stream = dict_results["stream"]
            self.windowed_evaluator = dict_results["windowed"]
            self.wallclock = dict_results["wallclock"]
            self.cpu_time = dict_results["cpu_time"]
            self.max_instances = dict_results["max_instances"]
            self.dict_results = dict_results
        else:
            if schema is None:
                raise ValueError("Schema must be initialised")
            self.schema = schema
            self.stream = stream
            self.windowed_evaluator = windowed_evaluator
            self.wallclock = wallclock
            self.cpu_time = cpu_time
            self.max_instances = max_instances

            self.dict_results = {
                "stream": self.stream,
                "schema": self.schema,
                "windowed": self.windowed_evaluator.metrics_dict()['windowed'],
                "wallclock": self.wallclock,
                "cpu_time": self.cpu_time,
                "max_instances": self.max_instances,
            }

    def __getitem__(self, key):
        return self.dict_results.get(key, None)

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, path=None):
        write_files(path=path, results=self)

    def learner(self):
        return self.learner

    def stream(self):
        return self.stream

    def schema(self):
        return self.stream.get_schema()

    def wallclock(self):
        return self.wallclock

    def cpu_time(self):
        return self.cpu_time

    def max_instances(self):
        return self.max_instances

    def metrics_per_window(self):
        return self.windowed_evaluator.metrics_per_window()

    def full_targets(self):
        return self.full_targets

    @property
    def windowed(self):
        return self.windowed_evaluator

    @property
    def results(self):
        return self.windowed

    # Classification Metrics
    def accuracy(self):
        if 'classifications correct (percent)' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.accuracy()
        else:
            raise ValueError("accuracy not exit")

    def kappa(self):
        if 'Kappa Statistic (percent)' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.kappa()
        else:
            raise ValueError("Kappa statistic not exit")

    def kappa_T(self):
        if 'Kappa Temporal Statistic (percent)' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.kappa_temporal()
        else:
            raise ValueError("Kappa temporal statistic not exit")

    def kappa_M(self):
        if 'Kappa M Statistic (percent)' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.kappa_M()
        else:
            raise ValueError("Kappa M-statistic not exit")

    def f1_score(self):
        if 'F1 Score (percent)' in self.windowed.metrics_dict().keys():
            return {'windowed f1 score':self.windowed_evaluator.metrics_per_window['F1 Score (percent)'].tolist()}
        else:
            raise ValueError("F1 score not exit")

    def precision(self):
        if 'Precision (percent)' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.metrics_dict()['Precision (percent)']
        else:
            raise ValueError("Precision not exit")

    def recall(self):
        if 'Recall (percent)' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.metrics_dict()['Recall (percent)']
        else:
            raise ValueError("Recall not exit")

    # Regression Metrics
    def MAE(self):
        if 'mean absolute error' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.MAE()
        else:
            raise ValueError("MAE not exit")

    def RMSE(self):
        if 'root mean squared error' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.RMSE()
        else:
            raise ValueError("RMSE not exit")

    def RMAE(self):
        if 'relative mean absolute error' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.RMAE()
        else:
            raise ValueError("RMAE not exit")

    def RRMSE(self):
        if 'relative root mean squared error' in self.windowed.metrics_dict().keys():
            return {'windowed RRMSE':self.windowed_evaluator.metrics_per_window()['relative root mean squared error'].tolist()}
        else:
            raise ValueError("RRMSE not exit")

    def R2(self):
        if 'coefficient of determination' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.R2()
        else:
            raise ValueError("R2 not exit")

    def adjusted_R2(self):
        if 'adjusted coefficient of determination' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.adjusted_R2()
        else:
            raise ValueError("adjusted R2 not exit")

    # Prediction Interval Metrics
    def coverage(self):
        if 'coverage' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.coverage()
        else:
            raise ValueError("Coverage not exit")

    def average_length(self):
        if 'average length' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.average_length()
        else:
            raise ValueError("Average length not exit")

    def NMPIW(self):
        if 'NMPIW' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.NMPIW()
        else:
            raise ValueError("NMPIW not exit")

    # Anomaly Metrics
    def auc(self):
        if 'AUC' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.metrics_dict()['auc']
        else:
            raise ValueError("AUC not exit")

    def s_auc(self):
        if 'sAUC' in self.windowed.metrics_dict().keys():
            return self.windowed_evaluator.metrics_dict()['sAUC']
        else:
            raise ValueError("sAUC not exit")


class PrequentialResults(Results):
    def __init__(self,
                 dict_results: dict = None,
                 schema: Schema = None,
                 stream: Stream = None,
                 wallclock: float = None,
                 cpu_time: float = None,
                 max_instances: int = None,
                 cumulative_evaluator=None,
                 windowed_evaluator=None,
                 store_y=False,
                 store_predictions=False,
                 ground_truth_y=None,
                 predictions=None,
                 prevent_saving_t_v_p: bool = False,

                 # for SSL
                 unlabeled = None,
                 unlabeled_ratio = None,
                 ):
        if dict_results is None:
            self.stream = stream
            self.schema = schema
            self.wallclock = wallclock
            self.cpu_time = cpu_time
            self.max_instances = max_instances
            if store_y:
                self.ground_truth_y = ground_truth_y
            if store_predictions:
                self.predictions = predictions
            self.cumulative_evaluator = cumulative_evaluator
            self.windowed_evaluator = windowed_evaluator

            # for SSL
            self.unlabeled = unlabeled
            self.unlabeled_ratio = unlabeled_ratio

            self.dict_results = {
                "stream": stream,
                "wallclock": wallclock,
                "cpu_time": cpu_time,
                "max_instances": max_instances,
                "cumulative": cumulative_evaluator,
                "windowed": windowed_evaluator,
                # for SSL
                "unlabeled": unlabeled,
                "unlabeled_ratio": unlabeled_ratio,

                "ground_truth_y": ground_truth_y,
                "predictions": predictions,
            }
        else:
            self.dict_results = dict_results
            self.schema = dict_results['stream'].get_schema()
            self.stream = dict_results['stream']
            self.wallclock = dict_results['wallclock']
            self.cpu_time = dict_results['cpu_time']
            self.max_instances = dict_results['max_instances']
            self.cumulative_evaluator = dict_results['cumulative']
            self.windowed_evaluator = dict_results['windowed']
            if store_y:
                self.ground_truth_y = dict_results['ground_truth_y']
            if store_predictions:
                self.predictions = dict_results['predictions']
            # for SSL
            if unlabeled is not None:
                self.unlabeled = dict_results['unlabeled']
            if unlabeled_ratio is not None:
                self.unlabeled_ratio = dict_results['unlabeled_ratio']

        self.prevent_saving_t_v_p = prevent_saving_t_v_p
    @property
    def cumulative(self):
        values = {
            'stream': self.dict_results['stream'],
            'wallclock': self.dict_results['wallclock'],
            'cpu_time': self.dict_results['cpu_time'],
            'max_instances': self.dict_results['max_instances'],
            'cumulative': self.dict_results['cumulative'],
        }
        return CumulativeResults(dict_results=values)

    @property
    def windowed(self):
        values = {
            'stream': self.dict_results['stream'],
            'wallclock': self.dict_results['wallclock'],
            'cpu_time': self.dict_results['cpu_time'],
            'max_instances': self.dict_results['max_instances'],
            'windowed': self.dict_results['windowed'],
        }
        return WindowedResults(dict_results=values)

    def __getitem__(self, key):
        return self.dict_results.get(key, None)

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, file_path: str | None = None):
        write_files(path=file_path, results=self)

    def learner(self):
        return self.learner

    def stream(self):
        return self.stream

    def schema(self):
        return self.stream.get_schema()

    def wallclock(self):
        return self.wallclock

    def cpu_time(self):
        return self.cpu_time

    def max_instances(self):
        return self.max_instances

    # @property
    def get_targets(self):
        return self.ground_truth_y

    # @property
    def get_predictions(self):
        return self.predictions

# Classification Metrics
    def accuracy(self):
        if 'classifications correct (percent)' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['classifications correct (percent)']
        else:
            raise ValueError("accuracy not exit")

    def kappa(self):
        if 'Kappa Statistic (percent)' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['Kappa Statistic (percent)']
        else:
            raise ValueError("Kappa statistic not exit")

    def kappa_T(self):
        if 'Kappa Temporal Statistic (percent)' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['Kappa Temporal Statistic (percent)']
        else:
            raise ValueError("Kappa temporal statistic not exit")

    def kappa_M(self):
        if 'Kappa M Statistic (percent)' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['Kappa M Statistic (percent)']
        else:
            raise ValueError("Kappa M-statistic not exit")

    def f1_score(self):
        if 'F1 Score (percent)' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['F1 Score (percent)']
        else:
            raise ValueError("F1 score not exit")

    def precision(self):
        if 'Precision (percent)' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['Precision (percent)']
        else:
            raise ValueError("Precision not exit")

    def recall(self):
        if 'Recall (percent)' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['Recall (percent)']
        else:
            raise ValueError("Recall not exit")

    # Regression Metrics
    def MAE(self):
        if 'mean absolute error' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['mean absolute error']
        else:
            raise ValueError("MAE not exit")

    def RMSE(self):
        if 'root mean squared error' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['root mean squared error']
        else:
            raise ValueError("RMSE not exit")

    def RMAE(self):
        if 'relative mean absolute error' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['relative mean absolute error']
        else:
            raise ValueError("RMAE not exit")

    def RRMSE(self):
        if 'relative root mean squared error' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['relative root mean squared error']
        else:
            raise ValueError("RRMSE not exit")

    def R2(self):
        if 'coefficient of determination' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['coefficient of determination']
        else:
            raise ValueError("R2 not exit")

    def adjusted_R2(self):
        if 'adjusted coefficient of determination' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['adjusted coefficient of determination']
        else:
            raise ValueError("adjusted R2 not exit")

    # Prediction Interval Metrics
    def coverage(self):
        if 'coverage' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['coverage']
        else:
            raise ValueError("Coverage not exit")

    def average_length(self):
        if 'average length' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['average length']
        else:
            raise ValueError("Average length not exit")

    def NMPIW(self):
        if 'NMPIW' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['NMPIW']
        else:
            raise ValueError("NMPIW not exit")

    # Anomaly Metrics
    def auc(self):
        if 'AUC' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['auc']
        else:
            raise ValueError("AUC not exit")

    def s_auc(self):
        if 'sAUC' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['sAUC']
        else:
            raise ValueError("sAUC not exit")

    # Semi-supervised Learning
    def unlabeled(self):
        if 'unlabeled' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['unlabeled']
        else:
            raise ValueError("unlabeled not exit")

    def unlabeled_ratio(self):
        if 'unlabeled_ratio' in self.cumulative.metrics_dict().keys():
            return self.cumulative_evaluator.metrics_dict()['unlabeled_ratio']
        else:
            raise ValueError("unlabeled_ratio not exit")

