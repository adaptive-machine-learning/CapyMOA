from jpype import JObject

import numpy as np
import torch

from capymoa.stream import Stream, Schema
from .stream import _init_moa_stream_and_create_moa_header,_add_instances_to_moa_stream
from capymoa.stream.instance import (
    LabeledInstance,
    RegressionInstance,
    _JavaLabeledInstance,
    _JavaRegressionInstance,
)

from moa.core import InstanceExample, Example, SerializeUtils
from com.yahoo.labs.samoa.instances import Instances

class PytorchStream(Stream):
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, dataset=None, enforce_regression=False):
        self.training_data = dataset
        # self.train_dataloader = DataLoader(self.training_data, batch_size=1, shuffle=False)
        self.current_instance_index = 0

        X, _ = self.training_data[0]
        X_numpy = torch.flatten(X).view(1,-1).detach().numpy()

        # enforce_regression = np.issubdtype(type(y[0]), np.double)

        self.__moa_stream_with_only_header, self.moa_header = _init_moa_stream_and_create_moa_header(
            number_of_instances=X_numpy.shape[0],
            feature_names=[f"attrib_{i}" for i in range(X_numpy.shape[1])],
            values_for_nominal_features= {},
            values_for_class_label=self.training_data.classes,
            dataset_name='PytorchDataset',
            target_attribute_name=None,
            enforce_regression=enforce_regression
        )


        self.schema = Schema(moa_header=self.moa_header, labels=self.training_data.classes)
        super().__init__(schema=self.schema, CLI=None, moa_stream=None)

    def has_more_instances(self):
        return len(self.training_data) > self.current_instance_index

    def next_instance(self):
        if self.has_more_instances():
            X, y = self.training_data[self.current_instance_index]
            tmp_moa_stream_with_only_header = JObject(SerializeUtils.copyObject(self.__moa_stream_with_only_header), Instances)
            _add_instances_to_moa_stream(tmp_moa_stream_with_only_header, self.moa_header, torch.flatten(X).view(1, -1).detach().numpy(), np.int32([y]))
            instance = tmp_moa_stream_with_only_header.instance(0)
            self.current_instance_index += 1 # increment counter for next call

            if self.schema.is_classification():
                return _JavaLabeledInstance(self.schema, InstanceExample(instance))
            elif self.schema.regression:
                return _JavaRegressionInstance(self.schema, InstanceExample(instance))
            else:
                raise ValueError(
                    "Unknown machine learning task must be a regression or "
                    "classification task"
                )
        else:
            # Return None if all instances have been read already.
            return None

    def get_schema(self):
        return self.schema

    def get_moa_stream(self):
        raise ValueError("Not a moa_stream, a numpy read file")

    def restart(self):
        self.current_instance_index = 0
