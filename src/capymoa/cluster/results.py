from capymoa.stream import Stream
import pandas as pd
import json
import csv
import os
from datetime import datetime

class PrequentialClusteringResults:
    def __init__(
            self,
            learner: str = None,
            stream: Stream = None,
            wallclock: float = None,
            cpu_time: float = None,
            max_instances: int = None,
            micro_cluster: list = None,
            macro_cluster: list = None,
            other_metrics=None,
    ):

        self.learner = learner
        self.stream = stream

        self._wallclock = wallclock
        self._cpu_time = cpu_time
        self._max_instances = max_instances
        self._micro_cluster = micro_cluster
        self._macro_cluster = macro_cluster
        self._other_metrics = other_metrics

    def __getitem__(self, key):
        # Check if the key is a method of this class or an attribute
        if hasattr(self, key):
            if callable(getattr(self, key)):
                # If it is a method, then invoke it and return its return value
                return getattr(self, key)()
            else:
                return getattr(self, key)

        # if hasattr(self.cumulative, key) and callable(getattr(self.cumulative, key)):
        #     return getattr(self.cumulative, key)()

        raise KeyError(f"Key {key} not found")

    def getattr(self, attribute):
        # Check if the attribute exists in the cumulative object
        if hasattr(self, attribute):
            return getattr(self, attribute)
        else:
            raise AttributeError(f"Attribute {attribute} not found")

    def wallclock(self):
        return self._wallclock
    def cpu_time(self):
        return self._cpu_time
    def max_instances(self):
        return self._max_instances
    def micros(self):
        return self._micro_cluster
    def macros(self):
        return self._macro_cluster

    def silhouette_past(self):
        if self._other_metrics is not None and 'silhouette_past'in self._other_metrics:
            return self._other_metrics['silhouette_past']
        else:
            raise ValueError("Silhouette past is not available")
    def silhouette_future(self):
        if self._other_metrics is not None and 'silhouette_future'in self._other_metrics:
            return self._other_metrics['silhouette_future']
        else:
            raise ValueError("Silhouette future is not available")
    def ssq_past(self):
        if self._other_metrics is not None and 'ssq_past'in self._other_metrics:
            return self._other_metrics['ssq_past']
        else:
            raise ValueError("SSQ past is not available")
    def ssq_future(self):
        if self._other_metrics is not None and 'ssq_future'in self._other_metrics:
            return self._other_metrics['ssq_future']
        else:
            raise ValueError("SSQ future is not available")
    def bss_past(self):
        if self._other_metrics is not None and 'bss_past'in self._other_metrics:
            return self._other_metrics['bss_past']
        else:
            raise ValueError("BSS past is not available")
    def bss_future(self):
        if self._other_metrics is not None and 'bss_future'in self._other_metrics:
            return self._other_metrics['bss_future']
        else:
            raise ValueError("BSS future is not available")