from abc import ABC, abstractmethod
import json, csv, os
from capymoa.stream import Stream, Schema
import pandas as pd
from typing import Union

_metric_name_mapping = {
    'accuracy': 'classifications correct (percent)',
    'kappa': 'Kappa Statistic (percent)',
    'kappa_t': 'Kappa Temporal Statistic (percent)',
    'kappa_m': 'Kappa M Statistic (percent)',
    'f1_score': 'F1 Score (percent)',
    'precision': 'Precision (percent)',
    'recall': 'Recall (percent)',

    'MAE': 'mean absolute error',
    'RMSE': 'root mean squared error',
    'RMAE': 'relative mean absolute error',
    'RRMSE': 'relative root mean squared error',
    'R2': "coefficient of determination",
    'Adjusted_R2': 'adjusted coefficient of determination',

    'coverage': 'coverage',
    'average_length': 'average length',
    'NMPIW': 'NMPIW',

    'auc': 'AUC',
    's_auc': 'sAUC',
}


def write_files(
        path: str = None,
        results=None,
        file_name: str = None,
        directory_name: str = None,
):
    if results is None:
        raise ValueError('nothing to write')

    path = path if path.endswith('/') else (path + '/')

    if isinstance(results, CumulativeResults):
        json_str = json.dumps(results.cumulative.metrics_dict())
        data = json.loads(json_str)
        with open(('./' if path is None else path) + ('/cumulative_results.csv' if file_name is None else file_name),
                  'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data.keys())
            writer.writerow(data.values())
    elif isinstance(results, WindowedResults):
        data = results.metrics_per_window()
        data.to_csv(('./' if path is None else path) + ('/windowed_results.csv' if file_name is None else file_name),
                    index=False)
    elif isinstance(results, dict):
        json_str = json.dumps(results)
        data = json.loads(json_str)
        with open(('./' if path is None else path) + ('/results.csv' if file_name is None else file_name), 'w',
                  newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data.keys())
            writer.writerow(data.values())
    elif isinstance(results, PrequentialResults):
        directory_name = 'prequential_results' if directory_name is None else directory_name
        if os.path.exists(path + '/' + directory_name):
            raise ValueError(f'directory {directory_name} exists, please use another name')
        else:
            os.makedirs(path + '/' + directory_name)

        write_files(path=path + '/' + directory_name, file_name=file_name, results=results.cumulative)
        write_files(path=path + '/' + directory_name, file_name=file_name, results=results.windowed)

        if results.prevent_saving_t_v_p is False:
            t_vs_p = {}
            if results.get_targets() is not None:
                t_vs_p['targets'] = results.get_targets()
            if results.get_predictions() is not None:
                t_vs_p['predictions'] = results.get_predictions()
            if len(t_vs_p) > 0:
                t_p = pd.DataFrame(t_vs_p)
                t_p.to_csv(('./' if path is None else path) + '/' + directory_name + '/targets_vs_predictions.csv',
                           index=False)
    else:
        raise ValueError('writing results to file is not supported for Type ' + str(type(results)) + ' yet')


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

##############################################################################
######################### Manner-wise Result Classes #########################
##############################################################################
class CumulativeResults(Results):
    def __init__(self, dict_results: dict = None, schema: Schema = None, stream: Stream = None,
                 cumulative_evaluator=None, wallclock: float = None, cpu_time: float = None, max_instances: int = None):
        super().__init__(dict_results, schema)
        if dict_results is not None:
            self._schema = dict_results["stream"].get_schema()
            self._stream = dict_results["stream"]
            self.cumulative_evaluator = dict_results["cumulative"]
            self._wallclock = dict_results["wallclock"]
            self._cpu_time = dict_results["cpu_time"]
            self._max_instances = dict_results["max_instances"]
            self.dict_results = dict_results

        else:
            if schema is None:
                raise ValueError("Schema must be initialised")
            self._schema = schema
            self._stream = stream
            self.cumulative_evaluator = cumulative_evaluator
            self._wallclock = wallclock
            self._cpu_time = cpu_time
            self._max_instances = max_instances

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

    def __setitem__(self, key, value):
        self.dict_results[key] = value

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, path=None):
        write_files(path=path, results=self)

    def learner(self):
        return self.learner

    def stream(self):
        return self._stream

    def schema(self):
        return self._stream.get_schema()

    def wallclock(self):
        return self._wallclock

    def cpu_time(self):
        return self._cpu_time

    def max_instances(self):
        return self._max_instances

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


class WindowedResults(Results):
    def __init__(self, dict_results: dict = None, schema: Schema = None, stream: Stream = None, windowed_evaluator=None,
                 wallclock: float = None, cpu_time: float = None, max_instances: int = None):
        super().__init__(dict_results, schema)
        if dict_results is not None:
            self._schema = dict_results["stream"].get_schema()
            self._stream = dict_results["stream"]
            self.windowed_evaluator = dict_results["windowed"]
            self._wallclock = dict_results["wallclock"]
            self._cpu_time = dict_results["cpu_time"]
            self._max_instances = dict_results["max_instances"]
            self.dict_results = dict_results
        else:
            if schema is None:
                raise ValueError("Schema must be initialised")
            self._schema = schema
            self._stream = stream
            self.windowed_evaluator = windowed_evaluator
            self._wallclock = wallclock
            self._cpu_time = cpu_time
            self._max_instances = max_instances

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

    def __setitem__(self, key, value):
        self.dict_results[key] = value


    def __str__(self):
        return str(self.dict_results)

    def to_file(self, path=None):
        write_files(path=path, results=self)

    def learner(self):
        return self.learner

    def stream(self):
        return self._stream

    def schema(self):
        return self._stream.get_schema()

    def wallclock(self):
        return self._wallclock

    def cpu_time(self):
        return self._cpu_time

    def max_instances(self):
        return self._max_instances

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


class PrequentialResults(Results):
    def __init__(self, dict_results: dict = None, schema: Schema = None, stream: Stream = None, wallclock: float = None,
                 cpu_time: float = None, max_instances: int = None, cumulative_evaluator=None, windowed_evaluator=None,
                 store_y=False, store_predictions=False, ground_truth_y=None, predictions=None,
                 prevent_saving_t_v_p: bool = False):
        super().__init__(dict_results, schema)
        if dict_results is None:
            self._stream = stream
            self._schema = schema
            self._wallclock = wallclock
            self._cpu_time = cpu_time
            self._max_instances = max_instances
            if store_y:
                self._ground_truth_y = ground_truth_y
            if store_predictions:
                self._predictions = predictions
            self.cumulative_evaluator = cumulative_evaluator
            self.windowed_evaluator = windowed_evaluator

            self.dict_results = {
                "stream": stream,
                "wallclock": wallclock,
                "cpu_time": cpu_time,
                "max_instances": max_instances,
                "cumulative": cumulative_evaluator,
                "windowed": windowed_evaluator,

                "ground_truth_y": ground_truth_y,
                "predictions": predictions,
            }
        else:
            self.dict_results = dict_results
            self._schema = dict_results['stream'].get_schema()
            self._stream = dict_results['stream']
            self._wallclock = dict_results['wallclock']
            self._cpu_time = dict_results['cpu_time']
            self._max_instances = dict_results['max_instances']
            self.cumulative_evaluator = dict_results['cumulative']
            self.windowed_evaluator = dict_results['windowed']
            if store_y:
                self._ground_truth_y = dict_results['ground_truth_y']
            if store_predictions:
                self._predictions = dict_results['predictions']

        self.prevent_saving_t_v_p = prevent_saving_t_v_p



    def __getitem__(self, key):
        return self.dict_results.get(key, None)

    def __setitem__(self, key, value):
        self.dict_results[key] = value

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

    def learner(self):
        return self.learner

    def stream(self):
        return self._stream

    def schema(self):
        return self._stream.get_schema()

    def wallclock(self):
        return self._wallclock

    def cpu_time(self):
        return self._cpu_time

    def max_instances(self):
        return self._max_instances

    def get_targets(self):
        return self.dict_results['ground_truth_y']

    def get_predictions(self):
        return self.dict_results['predictions']


class CumulativeSSLResults(Results):
    def __init__(self, dict_results: dict = None, schema: Schema = None, stream: Stream = None,
                 cumulative_evaluator=None, wallclock: float = None, cpu_time: float = None, max_instances: int = None,
                 unlabeled=None, unlabeled_ratio=None):
        super().__init__(dict_results, schema)
        if dict_results is not None:
            self._schema = dict_results["stream"].get_schema()
            self._stream = dict_results["stream"]
            self.cumulative_evaluator = dict_results["cumulative"]
            self._wallclock = dict_results["wallclock"]
            self._cpu_time = dict_results["cpu_time"]
            self._max_instances = dict_results["max_instances"]
            self.dict_results = dict_results
            # commented out because the test_then_train_ssl_evaluation is not fully implemented
            # self._unlabeled = dict_results["unlabeled"]
            # self._unlabeled_ratio = dict_results["unlabeled_ratio"]

        else:
            if schema is None:
                raise ValueError("Schema must be initialised")
            self._schema = schema
            self._stream = stream
            self.cumulative_evaluator = cumulative_evaluator
            self._wallclock = wallclock
            self._cpu_time = cpu_time
            self._max_instances = max_instances
            self._unlabeled = unlabeled
            self._unlabeled_ratio = unlabeled_ratio

            self.dict_results = {
                "stream": self.stream,
                "schema": self.schema,
                "cumulative": self.cumulative_evaluator.metrics_dict()['cumulative'],
                "wallclock": self.wallclock,
                "cpu_time": self.cpu_time,
                "max_instances": self.max_instances,
                "unlabeled": self._unlabeled,
                "unlabeled_ratio": self._unlabeled_ratio,
            }

    def __getitem__(self, key):
        return self.dict_results.get(key, None)

    def __setitem__(self, key, value):
        self.dict_results[key] = value

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, path=None):
        write_files(path=path, results=self)

    def learner(self):
        return self.learner

    def stream(self):
        return self._stream

    def schema(self):
        return self._stream.get_schema()

    def wallclock(self):
        return self._wallclock

    def cpu_time(self):
        return self._cpu_time

    def max_instances(self):
        return self._max_instances

    def full_targets(self):
        return self.full_targets

    def metrics_dict(self):
        return self.cumulative.metrics_dict()

    def unlabeled(self):
        return self._unlabeled

    def unlabeled_ratio(self):
        return self._unlabeled_ratio

    @property
    def cumulative(self):
        return self.cumulative_evaluator

    @property
    def results(self):
        return self.cumulative

class WindowedSSLResults(Results):
    def __init__(self, dict_results: dict = None, schema: Schema = None, stream: Stream = None, windowed_evaluator=None,
                 wallclock: float = None, cpu_time: float = None, max_instances: int = None, unlabeled=None, unlabeled_ratio=None):
        super().__init__(dict_results, schema)
        if dict_results is not None:
            self._schema = dict_results["stream"].get_schema()
            self._stream = dict_results["stream"]
            self.windowed_evaluator = dict_results["windowed"]
            self._wallclock = dict_results["wallclock"]
            self._cpu_time = dict_results["cpu_time"]
            self._max_instances = dict_results["max_instances"]
            self.dict_results = dict_results
            self._unlabeled = dict_results["unlabeled"]
            self._unlabeled_ratio = dict_results["unlabeled_ratio"]
        else:
            if schema is None:
                raise ValueError("Schema must be initialised")
            self._schema = schema
            self._stream = stream
            self.windowed_evaluator = windowed_evaluator
            self._wallclock = wallclock
            self._cpu_time = cpu_time
            self._max_instances = max_instances
            self._unlabeled = unlabeled
            self._unlabeled_ratio = unlabeled_ratio

            self.dict_results = {
                "stream": self.stream,
                "schema": self.schema,
                "windowed": self.windowed_evaluator.metrics_dict()['windowed'],
                "wallclock": self.wallclock,
                "cpu_time": self.cpu_time,
                "max_instances": self.max_instances,
                "unlabeled": self._unlabeled,
                "unlabeled_ratio": self._unlabeled_ratio,
            }

    def __getitem__(self, key):
        return self.dict_results.get(key, None)

    def __setitem__(self, key, value):
        self.dict_results[key] = value

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, path=None):
        write_files(path=path, results=self)

    def learner(self):
        return self.learner

    def stream(self):
        return self._stream

    def schema(self):
        return self._stream.get_schema()

    def wallclock(self):
        return self._wallclock

    def cpu_time(self):
        return self._cpu_time

    def max_instances(self):
        return self._max_instances

    def metrics_per_window(self):
        return self.windowed_evaluator.metrics_per_window()

    def full_targets(self):
        return self.full_targets

    def unlabeled(self):
        return self._unlabeled

    def unlabeled_ratio(self):
        return self._unlabeled_ratio
    @property
    def windowed(self):
        return self.windowed_evaluator

    @property
    def results(self):
        return self.windowed

class PrequentialSSLResults(Results):
    def __init__(self, dict_results: dict = None, schema: Schema = None, stream: Stream = None, wallclock: float = None,
                 cpu_time: float = None, max_instances: int = None, cumulative_evaluator=None, windowed_evaluator=None,
                 store_y=False, store_predictions=False, ground_truth_y=None, predictions=None,
                 prevent_saving_t_v_p: bool = False, unlabeled=None, unlabeled_ratio=None):
        super().__init__(dict_results, schema)
        if dict_results is None:
            self._stream = stream
            self._schema = schema
            self._wallclock = wallclock
            self._cpu_time = cpu_time
            self._max_instances = max_instances
            if store_y:
                self._ground_truth_y = ground_truth_y
            if store_predictions:
                self._predictions = predictions
            self.cumulative_evaluator = cumulative_evaluator
            self.windowed_evaluator = windowed_evaluator
            self._unlabeled = unlabeled,
            self._unlabeled_ratio = unlabeled_ratio,

            self.dict_results = {
                "stream": stream,
                "wallclock": wallclock,
                "cpu_time": cpu_time,
                "max_instances": max_instances,
                "cumulative": cumulative_evaluator,
                "windowed": windowed_evaluator,

                "ground_truth_y": ground_truth_y,
                "predictions": predictions,

                "unlabeled": unlabeled,
                "unlabeled_ratio": unlabeled_ratio,
            }
        else:
            self.dict_results = dict_results
            self._schema = dict_results['stream'].get_schema()
            self._stream = dict_results['stream']
            self._wallclock = dict_results['wallclock']
            self._cpu_time = dict_results['cpu_time']
            self._max_instances = dict_results['max_instances']
            self.cumulative_evaluator = dict_results['cumulative']
            self.windowed_evaluator = dict_results['windowed']
            if store_y:
                self._ground_truth_y = dict_results['ground_truth_y']
            if store_predictions:
                self._predictions = dict_results['predictions']

            if unlabeled is not None:
                self._unlabeled = dict_results['unlabeled']
            if unlabeled_ratio is not None:
                self._unlabeled_ratio = dict_results['unlabeled_ratio']

        self.prevent_saving_t_v_p = prevent_saving_t_v_p



    def __getitem__(self, key):
        return self.dict_results.get(key, None)

    def __setitem__(self, key, value):
        self.dict_results[key] = value

    def __str__(self):
        return str(self.dict_results)


    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

    def learner(self):
        return self.learner

    def stream(self):
        return self._stream

    def schema(self):
        return self._stream.get_schema()

    def wallclock(self):
        return self._wallclock

    def cpu_time(self):
        return self._cpu_time

    def max_instances(self):
        return self._max_instances

    def unlabeled(self):
        return self._unlabeled

    def unlabeled_ratio(self):
        return self._unlabeled_ratio

    def get_targets(self):
        return self.dict_results['ground_truth_y']

    def get_predictions(self):
        return self.dict_results['predictions']


##############################################################################
########################## Task-wise Result Classes ##########################
##############################################################################
class ClassificationResults():
    def __init__(self,
                 cumulative_evaluator=None,
                 windowed_evaluator=None,
                 ):
        self.cumulative_evaluator = cumulative_evaluator
        self.windowed_evaluator = windowed_evaluator

    def accuracy(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['accuracy']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['accuracy']].tolist()
        else:
            raise ValueError('no results found')

    def kappa(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['kappa']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['kappa']].tolist()
        else:
            raise ValueError('no results found')

    def kappa_t(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['kappa_t']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['kappa_t']].tolist()
        else:
            raise ValueError('no results found')

    def kappa_m(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['kappa_m']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['kappa_m']].tolist()
        else:
            raise ValueError('no results found')

    def f1_score(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['f1_score']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['f1_score']].tolist()
        else:
            raise ValueError('no results found')

    def precision(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['precision']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['precision']].tolist()
        else:
            raise ValueError('no results found')

    def recall(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['recall']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['recall']].tolist()
        else:
            raise ValueError('no results found')


class RegressionResults():
    def __init__(self,
                 cumulative_evaluator=None,
                 windowed_evaluator=None,
                 ):
        self.cumulative_evaluator = cumulative_evaluator
        self.windowed_evaluator = windowed_evaluator

    def mae(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['MAE']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['MAE']].tolist()
        else:
            raise ValueError('no results found')

    def rmse(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['RMSE']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['RMSE']].tolist()
        else:
            raise ValueError('no results found')

    def rmae(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['RMAE']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['RMAE']].tolist()
        else:
            raise ValueError('no results found')

    def rrmse(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['RRMSE']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['RRMSE']].tolist()
        else:
            raise ValueError('no results found')

    def r2(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['R2']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['R2']].tolist()
        else:
            raise ValueError('no results found')

    def adjusted_r2(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['Adjusted_R2']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['Adjusted_R2']].tolist()
        else:
            raise ValueError('no results found')


class PredictionIntervalResults(RegressionResults):
    def __init__(self, cumulative_evaluator=None, windowed_evaluator=None):
        super().__init__(cumulative_evaluator, windowed_evaluator)
        self.cumulative_evaluator = cumulative_evaluator
        self.windowed_evaluator = windowed_evaluator

    def coverage(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['coverage']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['coverage']].tolist()
        else:
            raise ValueError('no results found')

    def average_length(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['average_length']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['average_length']].tolist()
        else:
            raise ValueError('no results found')

    def NMPIW(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['NMPIW']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['NMPIW']].tolist()
        else:
            raise ValueError('no results found')


class AnomalyDetectionResults():
    def __init__(
            self,
            cumulative_evaluator=None,
            windowed_evaluator=None
    ):
        self.cumulative_evaluator = cumulative_evaluator
        self.windowed_evaluator = windowed_evaluator

    def auc(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['auc']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['auc']].tolist()
        else:
            raise ValueError('no results found')

    def s_auc(self):
        if self.cumulative_evaluator is not None:
            return self.cumulative_evaluator.metrics_dict()[_metric_name_mapping['s_auc']]
        elif self.windowed_evaluator is not None:
            return self.windowed_evaluator.metrics_per_window()[_metric_name_mapping['s_auc']].tolist()
        else:
            raise ValueError('no results found')

##############################################################################
########################## Specific Result Classes ###########################
##############################################################################

# Classification
class CumulativeClassificationResults(CumulativeResults, ClassificationResults):
    def __init__(self, dict_results: dict = None):
        CumulativeResults.__init__(self, dict_results)
        ClassificationResults.__init__(self, cumulative_evaluator = dict_results['cumulative'], windowed_evaluator = None)

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

class WindowedClassificationResults(WindowedResults, ClassificationResults):
    def __init__(self, dict_results: dict = None):

        WindowedResults.__init__(self, dict_results)
        ClassificationResults.__init__(self, cumulative_evaluator = None, windowed_evaluator = dict_results['windowed'])

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

class PrequentialClassificationResults(PrequentialResults, ClassificationResults):
    def __init__(self, dict_results: dict = None):

        PrequentialResults.__init__(self, dict_results = dict_results)
        ClassificationResults.__init__(self, cumulative_evaluator=dict_results['cumulative'], windowed_evaluator=dict_results['windowed'])

    def __str__(self):
        return str(self.dict_results)

    def __getitem__(self, key):
        if key == 'cumulative':
            return self.cumulative
        elif key == 'windowed':
            return self.windowed
        else:
            return self.dict_results.get(key, None)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

    @property
    def cumulative(self):
        values = {
            'stream': self.dict_results['stream'],
            'wallclock': self.dict_results['wallclock'],
            'cpu_time': self.dict_results['cpu_time'],
            'max_instances': self.dict_results['max_instances'],
            'cumulative': self.dict_results['cumulative'],
        }
        return CumulativeClassificationResults(dict_results=values)

    @property
    def windowed(self):
        values = {
            'stream': self.dict_results['stream'],
            'wallclock': self.dict_results['wallclock'],
            'cpu_time': self.dict_results['cpu_time'],
            'max_instances': self.dict_results['max_instances'],
            'windowed': self.dict_results['windowed'],
        }
        return WindowedClassificationResults(dict_results=values)

# Regression
class CumulativeRegressionResults(CumulativeResults, RegressionResults):
    def __init__(self, dict_results: dict = None):
        CumulativeResults.__init__(self, dict_results)
        RegressionResults.__init__(self, cumulative_evaluator = dict_results['cumulative'], windowed_evaluator = None)

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

class WindowedRegressionResults(WindowedResults, RegressionResults):
    def __init__(self, dict_results: dict = None):

        WindowedResults.__init__(self, dict_results)
        RegressionResults.__init__(self, cumulative_evaluator = None, windowed_evaluator = dict_results['windowed'])

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

class PrequentialRegressionResults(PrequentialResults, RegressionResults):
    def __init__(self, dict_results: dict = None):

        PrequentialResults.__init__(self, dict_results = dict_results)
        RegressionResults.__init__(self, cumulative_evaluator=dict_results['cumulative'], windowed_evaluator=dict_results['windowed'])

    def __str__(self):
        return str(self.dict_results)

    def __getitem__(self, key):
        if key == 'cumulative':
            return self.cumulative
        elif key == 'windowed':
            return self.windowed
        else:
            return self.dict_results.get(key, None)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

    @property
    def cumulative(self):
        values = {
            'stream': self.dict_results['stream'],
            'wallclock': self.dict_results['wallclock'],
            'cpu_time': self.dict_results['cpu_time'],
            'max_instances': self.dict_results['max_instances'],
            'cumulative': self.dict_results['cumulative'],
        }
        return CumulativeRegressionResults(dict_results=values)

    @property
    def windowed(self):
        values = {
            'stream': self.dict_results['stream'],
            'wallclock': self.dict_results['wallclock'],
            'cpu_time': self.dict_results['cpu_time'],
            'max_instances': self.dict_results['max_instances'],
            'windowed': self.dict_results['windowed'],
        }
        return WindowedRegressionResults(dict_results=values)

# Prediction Interval
class CumulativePredictionIntervalResults(CumulativeResults, PredictionIntervalResults):
    def __init__(self, dict_results: dict = None,):
        CumulativeResults.__init__(self, dict_results)
        PredictionIntervalResults.__init__(self, cumulative_evaluator = dict_results['cumulative'], windowed_evaluator = None)

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

class WindowedPredictionIntervalResults(WindowedResults, PredictionIntervalResults):
    def __init__(self, dict_results: dict = None):

        WindowedResults.__init__(self, dict_results)
        PredictionIntervalResults.__init__(self, cumulative_evaluator = None, windowed_evaluator = dict_results['windowed'])

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

class PrequentialPredictionIntervalResults(PrequentialResults, PredictionIntervalResults):
    def __init__(self, dict_results: dict = None):

        PrequentialResults.__init__(self, dict_results = dict_results)
        PredictionIntervalResults.__init__(self, cumulative_evaluator=dict_results['cumulative'], windowed_evaluator=dict_results['windowed'])

    def __str__(self):
        return str(self.dict_results)

    def __getitem__(self, key):
        if key == 'cumulative':
            return self.cumulative
        elif key == 'windowed':
            return self.windowed
        else:
            return self.dict_results.get(key, None)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

    @property
    def cumulative(self):
        values = {
            'stream': self.dict_results['stream'],
            'wallclock': self.dict_results['wallclock'],
            'cpu_time': self.dict_results['cpu_time'],
            'max_instances': self.dict_results['max_instances'],
            'cumulative': self.dict_results['cumulative'],
        }
        return CumulativePredictionIntervalResults(dict_results=values)

    @property
    def windowed(self):
        values = {
            'stream': self.dict_results['stream'],
            'wallclock': self.dict_results['wallclock'],
            'cpu_time': self.dict_results['cpu_time'],
            'max_instances': self.dict_results['max_instances'],
            'windowed': self.dict_results['windowed'],
        }
        return WindowedPredictionIntervalResults(dict_results=values)

# Anomaly Detection
class CumulativeAnomalyDetectionResults(CumulativeResults, AnomalyDetectionResults):
    def __init__(self, dict_results: dict = None):
        CumulativeResults.__init__(self, dict_results)
        AnomalyDetectionResults.__init__(self, cumulative_evaluator = dict_results['cumulative'])

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

class WindowedAnomalyDetectionResults(WindowedResults, AnomalyDetectionResults):
    def __init__(self, dict_results: dict = None):

        WindowedResults.__init__(self, dict_results)
        AnomalyDetectionResults.__init__(self, cumulative_evaluator = None, windowed_evaluator = dict_results['windowed'])

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

class PrequentialAnomalyDetectionResults(PrequentialResults, AnomalyDetectionResults):
    def __init__(self, dict_results: dict = None):

        PrequentialResults.__init__(self, dict_results = dict_results)
        AnomalyDetectionResults.__init__(self, cumulative_evaluator=dict_results['cumulative'], windowed_evaluator=dict_results['windowed'])

    def __str__(self):
        return str(self.dict_results)

    def __getitem__(self, key):
        if key == 'cumulative':
            return self.cumulative
        elif key == 'windowed':
            return self.windowed
        else:
            return self.dict_results.get(key, None)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

    @property
    def cumulative(self):
        values = {
            'stream': self.dict_results['stream'],
            'wallclock': self.dict_results['wallclock'],
            'cpu_time': self.dict_results['cpu_time'],
            'max_instances': self.dict_results['max_instances'],
            'cumulative': self.dict_results['cumulative'],
        }
        return CumulativeAnomalyDetectionResults(dict_results=values)

    @property
    def windowed(self):
        values = {
            'stream': self.dict_results['stream'],
            'wallclock': self.dict_results['wallclock'],
            'cpu_time': self.dict_results['cpu_time'],
            'max_instances': self.dict_results['max_instances'],
            'windowed': self.dict_results['windowed'],
        }
        return WindowedAnomalyDetectionResults(dict_results=values)


class CumulativeClassificationSSLResults(CumulativeSSLResults, ClassificationResults):
    def __init__(self, dict_results: dict = None):
        CumulativeSSLResults.__init__(self, dict_results)
        ClassificationResults.__init__(self, cumulative_evaluator = dict_results['cumulative'], windowed_evaluator = None)

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

class WindowedClassificationSSLResults(WindowedSSLResults, ClassificationResults):
    def __init__(self, dict_results: dict = None):

        WindowedSSLResults.__init__(self, dict_results)
        ClassificationResults.__init__(self, cumulative_evaluator = None, windowed_evaluator = dict_results['windowed'])

    def __str__(self):
        return str(self.dict_results)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

class PrequentialClassificationSSLResults(PrequentialSSLResults, ClassificationResults):
    def __init__(self, dict_results: dict = None):

        PrequentialSSLResults.__init__(self, dict_results = dict_results)
        ClassificationResults.__init__(self, cumulative_evaluator=dict_results['cumulative'], windowed_evaluator=dict_results['windowed'])

    def __str__(self):
        return str(self.dict_results)

    def __getitem__(self, key):
        if key == 'cumulative':
            return self.cumulative
        elif key == 'windowed':
            return self.windowed
        else:
            return self.dict_results.get(key, None)

    def to_file(self, file_path: str = None):
        write_files(path=file_path, results=self)

    @property
    def cumulative(self):
        values = {
            'stream': self.dict_results['stream'],
            'wallclock': self.dict_results['wallclock'],
            'cpu_time': self.dict_results['cpu_time'],
            'max_instances': self.dict_results['max_instances'],
            'cumulative': self.dict_results['cumulative'],
        }
        return CumulativeClassificationResults(dict_results=values)

    @property
    def windowed(self):
        values = {
            'stream': self.dict_results['stream'],
            'wallclock': self.dict_results['wallclock'],
            'cpu_time': self.dict_results['cpu_time'],
            'max_instances': self.dict_results['max_instances'],
            'windowed': self.dict_results['windowed'],
        }
        return WindowedClassificationResults(dict_results=values)