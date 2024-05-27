from __future__ import annotations

from typing import Optional, List

from capymoa.base import Classifier, Regressor
from capymoa.instance import LabeledInstance, Instance, RegressionInstance
from capymoa.stream.preprocessing.transformer import Transformer
from capymoa.type_alias import LabelProbabilities, LabelIndex, TargetValue


class BasePipeline:
    def __init__(self, transformers: List[Transformer] | None = None):
        self.elements: List[Transformer] = [] if transformers is None else transformers

    def add_transformer(self, new_element: Transformer):
        assert isinstance(new_element, Transformer), "Please provide a Transformer object"
        self.elements.append(new_element)

    def transform(self, instance: Instance) -> Instance:
        inst = instance
        for i, element in enumerate(self.elements):
            inst = element.transform_instance(inst)
        return inst

    def __str__(self):
        s = ""
        for i, transformer in enumerate(self.elements):
            s += str(transformer)
            if i == len(self.elements) - 1:
                break
            s += " | "
        return s


class ClassifierPipeline(BasePipeline, Classifier):

    def __init__(self, transformers: List[Transformer] | None = None, learner: Classifier | None = None):
        super(ClassifierPipeline, self).__init__(transformers)
        self.learner = learner

    def train(self, instance: LabeledInstance):
        instance = self.transform(instance)
        self.learner.train(instance)

    def predict(self, instance: Instance) -> Optional[LabelIndex]:
        instance = self.transform(instance)
        return self.learner.predict(instance)

    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        instance = self.transform(instance)
        return self.learner.predict_proba(instance)

    def set_learner(self, learner: Classifier | Regressor):
        self.learner = learner

    def __str__(self):
        s = ""
        for i, transformer in enumerate(self.elements):
            s += str(transformer)
            s += " | "
        return s + str(self.learner)


class RegressorPipeline(BasePipeline, Regressor):
    def __init__(self, transformers: List[Transformer] | None = None, learner: Regressor | None = None):
        super(RegressorPipeline, self).__init__(transformers)
        self.learner = learner

    def train(self, instance: RegressionInstance):
        instance = self.transform(instance)
        self.learner.train(instance)

    def predict(self, instance: RegressionInstance) -> TargetValue:
        instance = self.transform(instance)
        return self.learner.predict(instance)

    def set_learner(self, learner: Classifier | Regressor):
        self.learner = learner

    def __str__(self):
        s = ""
        for i, transformer in enumerate(self.elements):
            s += str(transformer)
            s += " | "
        return s + str(self.learner)
