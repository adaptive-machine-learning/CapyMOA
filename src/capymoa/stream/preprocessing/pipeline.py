from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from capymoa.base import Classifier, Regressor
from capymoa.core import (
    Instance,
    LabeledInstance,
    LabelIndex,
    LabelProbabilities,
    RegressionInstance,
    TargetValue,
)
from capymoa.drift.base_detector import BaseDriftDetector

from .._stream import Schema
from .transformer import Transformer


class PipelineElement(Protocol):
    """
    The basic pipeline building block
    """

    @abstractmethod
    def pass_forward(self, instance: Instance) -> Instance:
        raise NotImplementedError

    @abstractmethod
    def pass_forward_predict(
        self, instance: Instance, prediction=None
    ) -> tuple[Instance, Any]:
        raise NotImplementedError

    def get_schema(self) -> Schema | None:
        """Return the schema of instances leaving this element.

        Returns ``None`` when the element neither knows nor alters the schema --
        a drift detector, for instance. The default is ``None`` so that existing
        :class:`PipelineElement` implementations keep working unchanged.
        """
        return None

    def get_input_schema(self) -> Schema | None:
        """Return the schema of instances this element expects to receive.

        Defaults to :meth:`get_schema`, which is correct for every element that
        does not alter the attribute set.
        """
        return self.get_schema()

    @abstractmethod
    def __str__(self):
        raise NotImplementedError


class ClassifierPipelineElement(PipelineElement):
    """
    Pipeline element that wraps around a classifier
    """

    def __init__(self, learner: Classifier):
        """__init__

        Initializes the pipeline element with a classifier.

        Parameters
        ----------
        learner: Classifier
            The classifier associated with this pipeline element.

        """
        self.learner = learner

    def pass_forward(self, instance: Instance) -> Instance:
        """pass_forward

        Trains the learner on the provided instance; then returns the instance.

        Parameters
        ----------
        instance: Instance
            An instance to train the learner

        Returns
        -------
        Instance
            The instance that was provided to the function

        """
        self.learner.train(instance)
        return instance

    def pass_forward_predict(
        self, instance: Instance, prediction: Any = None
    ) -> tuple[Instance, Any]:
        """pass_forward_predict

        Trains the learner on the provided instance; then returns the instance.

        Parameters
        ----------
        instance: Instance
            An instance to train the learner
        prediction: Any
            Most likely None, but could be anything in principle

        Returns
        -------
        Tuple[Instance, Any]
            The transformed instance and the prediction of the classifier

        """
        return instance, self.learner.predict(instance)

    def get_schema(self) -> Schema | None:
        """Return the schema the wrapped classifier expects; learners do not alter it."""
        return getattr(self.learner, "schema", None)

    def __str__(self):
        return f"PE({self.learner!s})"


class RegressorPipelineElement(PipelineElement):
    """
    Pipeline element that wraps around a regressor
    """

    def __init__(self, learner: Regressor):
        """__init__

        Initializes the pipeline element with a regressor.

        Parameters
        ----------
        learner: Regressor
            The regressor associated with this pipeline element.

        """
        self.learner = learner

    def pass_forward(self, instance: Instance) -> Instance:
        """pass_forward

        Trains the learner on the provided instance; then returns the instance.

        Parameters
        ----------
        instance: Instance
            An instance to train the learner

        Returns
        -------
        Instance
            The instance that was provided to the function

        """
        self.learner.train(instance)
        return instance

    def pass_forward_predict(
        self, instance: Instance, prediction=None
    ) -> tuple[Instance, Any]:
        """pass_forward_predict

        Trains the learner on the provided instance; then returns the instance.

        Parameters
        ----------
        instance: Instance
            An instance to train the learner
        prediction: Any
            Most likely None, but could be anything in principle

        Returns
        -------
        Tuple[Instance, Any]
            The transformed instance and the prediction of the regressor

        """
        return instance, self.learner.predict(instance)

    def get_schema(self) -> Schema | None:
        """Return the schema the wrapped regressor expects; learners do not alter it."""
        return getattr(self.learner, "schema", None)

    def __str__(self):
        return f"PE({self.learner!s})"


class TransformerPipelineElement(PipelineElement):
    """
    Pipeline element that wraps around a transformer
    """

    def __init__(self, transformer: Transformer):
        """__init__

        Initializes the pipeline element with a transformer.

        Parameters
        ----------
        transformer: Transformer
            The transformer associated with this pipeline element.

        """
        self.transformer = transformer

    def pass_forward(self, instance: Instance) -> Instance:
        """pass_forward

        Transforms and returns the provided instance.

        Parameters
        ----------
        instance: Instance
            The input instance

        Returns
        -------
        instance: Instance
            The transformed instance

        """
        return self.transformer.transform_instance(instance)

    def pass_forward_predict(
        self, instance: Instance, prediction: Any = None
    ) -> tuple[Instance, Any]:
        """pass_forward_predict

        Transforms and returns the provided instance. Also returns the prediction that was provided.

        Parameters
        ----------
        instance: Instance
            The input instance
        prediction: Any
            Most likely None, but could be anything.

        Returns
        -------
        Tuple[Instance, Any]
            The transformed instance and the prediction that was provided

        """
        return self.transformer.transform_instance(instance), prediction

    def get_schema(self) -> Schema | None:
        """Return the schema of instances leaving the wrapped transformer."""
        return self.transformer.get_schema()

    def get_input_schema(self) -> Schema | None:
        """Return the schema the wrapped transformer expects to receive."""
        return self.transformer.get_input_schema()

    def __str__(self):
        return f"PE({self.transformer!s})"


class DriftDetectorPipelineElement(PipelineElement):
    """
    Pipeline element that wraps around a drift detector
    """

    def __init__(
        self,
        drift_detector: BaseDriftDetector,
        prepare_drift_detector_input_func: Callable,
    ):
        """__init__

        Initializes the pipeline element with a drift detector.

        Parameters
        ----------
        drift_detector: BaseDriftDetector
            The drift detector that associated with the pipeline element
        prepare_drift_detector_input_func: Callable
            The function that prepares the input of the drift detector.
            The function signature should start with the instance and the prediction.
            E.g., prediction_is_correct(instance, pred). The output of that function gets passed to the drift detector

        """
        self.drift_detector = drift_detector
        self.prepare_drift_detector_input_func = prepare_drift_detector_input_func

    def pass_forward(self, instance: Instance) -> Instance:
        """pass_forward

        Simply returns the instance. The drift detector gets updated in pass_forward_predict.

        Parameters
        ----------
        instance: Instance
            The instance

        Returns
        -------
        Instance
            The instance that was provided as input

        """
        return instance

    def pass_forward_predict(
        self, instance: Instance, prediction: Any = None
    ) -> tuple[Instance, Any]:
        """pass_forward_predict

        Updates the drift detector; returns the instance and the prediction that were provided to the function

        Parameters
        ----------
        instance: Instance:
            The instance
        prediction: Any
            The prediction from the previous pipeline steps.
            This can be None (e.g., when monitoring the the instance),
            an integer (e.g., when monitoring a classifier),
            or a float (when monitoring a regressor).
            It can also be anything else, but it must be compatible with prepare_drift_detector_input_func

        Returns
        -------
        Tuple[Instance, Any]
            The instance and prediction that were provided as input

        """
        drift_detector_input = self.prepare_drift_detector_input_func(
            instance, prediction
        )
        self.drift_detector.add_element(drift_detector_input)
        return instance, prediction

    def __str__(self):
        return f"PE({self.drift_detector!s})"


class BasePipeline(PipelineElement):
    """
    The base class for other types of pipelines. Supports transformers and drift detectors.
    """

    def __init__(
        self,
        pipeline_elements: list[PipelineElement] | None = None,
        schema: Schema | None = None,
        random_seed: int = 1,
        validate_schema: bool = True,
    ):
        """__init__

        Initializes the base pipeline with a list of pipeline elements.

        Parameters
        ----------
        pipeline_elements: List[PipelineElement]
            The elements the pipeline consists of
        schema: Optional[Schema]
            The schema of instances entering the pipeline. Normally left unset,
            in which case it is taken from the first element that knows one.
        random_seed: int
            Seed reported to satisfy the learner interface. The pipeline does
            not draw from it; its elements carry their own seeds.
        validate_schema: bool
            If True, adding an element whose schema is incompatible with the
            schema leaving the pipeline raises a ValueError.

        """
        self._input_schema = schema
        self.random_seed = random_seed
        self.validate_schema = validate_schema
        # Added one at a time so that elements passed here are checked against
        # each other exactly as elements appended later are.
        self.elements: list[PipelineElement] = []
        for element in pipeline_elements or []:
            self.add_pipeline_element(element)

    @property
    def schema(self) -> Schema | None:
        """The schema of instances the pipeline consumes.

        This is the *input* schema, because that is what
        :class:`capymoa.base.Classifier` and :class:`capymoa.base.Regressor`
        mean by ``schema``: the instances handed to ``train`` and ``predict``.
        What the pipeline emits downstream may differ, and is reported by
        :meth:`get_schema`.

        Defined as a property so that it keeps up with elements added after
        construction.
        """
        return self.get_input_schema()

    @schema.setter
    def schema(self, value: Schema | None) -> None:
        """Set the schema of instances entering the pipeline.

        Validated against the first element the same way :meth:`add_pipeline_element`
        validates a new element against what the pipeline currently emits --
        otherwise this setter would be an unchecked back door around the
        compatibility check ``add_pipeline_element`` enforces.
        """
        if self.validate_schema and value is not None and self.elements:
            incoming = self.elements[0].get_input_schema()
            if incoming is not None and not incoming.is_compatible_with(value):
                differences = "; ".join(incoming.describe_difference(value))
                raise ValueError(
                    f"Cannot set schema on the pipeline: {self.elements[0]} "
                    f"expects a different schema ({differences}). Pass "
                    "validate_schema=False to the pipeline to skip this check."
                )
        self._input_schema = value

    def get_input_schema(self) -> Schema | None:
        """Return the schema of instances entering the pipeline.

        This is the schema of the first element that knows one, unless it was
        given explicitly at construction.
        """
        if self._input_schema is not None:
            return self._input_schema
        for element in self.elements:
            schema = element.get_input_schema()
            if schema is not None:
                return schema
        return None

    def get_schema(self) -> Schema | None:
        """Return the schema of instances leaving the pipeline.

        This is the schema of the last element that knows one, so a pipeline
        nested inside another reports what its own last element produces. Falls
        back to the input schema when no element alters it, and is ``None`` for
        an empty pipeline with no declared schema.
        """
        for element in reversed(self.elements):
            schema = element.get_schema()
            if schema is not None:
                return schema
        return self._input_schema

    def _check_schema_compatibility(self, element: PipelineElement) -> None:
        """Raise if ``element`` cannot consume what the pipeline currently emits.

        Silently accepts the case where either side does not know its schema --
        an unknown schema is not evidence of a mismatch.
        """
        if not self.validate_schema:
            return
        outgoing = self.get_schema()
        incoming = element.get_input_schema()
        if outgoing is None or incoming is None:
            return
        if incoming.is_compatible_with(outgoing):
            return
        differences = "; ".join(incoming.describe_difference(outgoing))
        raise ValueError(
            f"Cannot add {element} to the pipeline: it expects a different "
            f"schema than the pipeline produces ({differences}). Pass "
            "validate_schema=False to the pipeline to skip this check."
        )

    def add_pipeline_element(self, element: PipelineElement):
        """add_pipeline_element

        Adds the provided pipeline element to the end of the pipeline

        Parameters
        ----------
        element: PipelineElement
            The element to add to the pipeline

        Returns
        -------
        BasePipeline
            self

        Raises
        ------
        ValueError
            If the element's schema is incompatible with the schema currently
            leaving the pipeline and ``validate_schema`` is enabled.

        """
        self._check_schema_compatibility(element)
        self.elements.append(element)
        return self

    def add_transformer(self, transformer: Transformer):
        """add_transformer

        Adds a transformer to the end of the current pipeline

        Parameters
        ----------
        transformer: Transformer
            The transformer to add

        Returns
        -------
        BasePipeline
            self

        """
        assert isinstance(transformer, Transformer), (
            "Please provide a Transformer object"
        )
        return self.add_pipeline_element(TransformerPipelineElement(transformer))

    def add_drift_detector(
        self, drift_detector: BaseDriftDetector, get_drift_detector_input_func: Callable
    ):
        """add_drift_detector

        Adds a drift detector to the end of the current pipeline

        Parameters
        ----------
        drift_detector: BaseDriftDetector
            The drift_detector to add
        get_drift_detector_input_func: Callable
            The function that prepares the input of the drift detector.
            The function signature should start with the instance and the prediction.
            E.g., prediction_is_correct(instance, pred). The output of that function gets passed to the drift detector

        Returns
        -------
        BasePipeline
            self

        """
        assert isinstance(drift_detector, BaseDriftDetector)
        return self.add_pipeline_element(
            DriftDetectorPipelineElement(drift_detector, get_drift_detector_input_func)
        )

    def pass_forward(self, instance: Instance) -> Instance:
        """pass_forward

        Passes the instance through the pipeline and returns it.
        This transforms the instance depending on the transformers in the pipeline

        Parameters
        ----------
        instance: Instance
            The instance

        Returns
        -------
        Instance
            The instance that exits the pipeline

        """
        inst = instance
        for i, element in enumerate(self.elements):
            inst = element.pass_forward(inst)
        return inst

    def pass_forward_predict(
        self, instance: Instance, prediction: Any = None
    ) -> tuple[Instance, Any]:
        """pass_forward_predict

        Passes the instance through the pipeline and returns it. Also returns the prediction of the pipeline.

        Parameters
        ----------
        instance: Instance
            The input instance
        prediction: Any
            The prediction passed to the pipeline.
            This can be useful to, e.g., set up a change detection pipeline after the prediction pipeline

        Returns
        -------
        Tuple[Instance, Any]
            The instance that exits the pipeline and the prediction that exits the pipeline.
            In the case of a BasePipeline, this is most likely the prediction that was given to the function

        """
        inst = instance
        pred = prediction
        for i, element in enumerate(self.elements):
            inst, pred = element.pass_forward_predict(inst, pred)
        return inst, pred

    def __str__(self):
        return " | ".join(str(element) for element in self.elements)


class ClassifierPipeline(BasePipeline, Classifier):
    """
    Classifier pipeline that (in addition to the functionality of BasePipeline) also acts as a classifier.
    """

    def add_classifier(self, classifier: Classifier):
        """add_classifier

        Adds a classifier to the end of the current pipeline

        Parameters
        ----------
        classifier: Classifier
            The classifier to add to the pipeline

        Returns
        -------
        ClassifierPipeline
            self

        """
        assert isinstance(classifier, Classifier), "Please provide a classifier object"
        return self.add_pipeline_element(ClassifierPipelineElement(classifier))

    def train(self, instance: LabeledInstance):
        """train

        The train function of the Classifier. Calls pass_forward internally.

        Parameters
        ----------
        instance: LabeledInstance
            The instance to train on

        """
        self.pass_forward(instance)
        return self

    def predict(self, instance: Instance) -> LabelIndex | None:
        """predict

        The predict function of the classifier.
        Calls pass_forward_predict internally and returns the prediction.

        Parameters
        ----------
        instance: Instance
            The instance to predict

        Returns
        -------
        Optional[LabelIndex]
            The prediction of the pipeline

        """
        _inst, pred = self.pass_forward_predict(instance)
        return pred

    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        # TODO: Discuss how handle this
        raise NotImplementedError


class RegressorPipeline(BasePipeline, Regressor):
    """
    Regressor pipeline that (in addition to the functionality of BasePipeline) also acts as a regressor.
    """

    def add_regressor(self, regressor: Regressor):
        """add_regressor

        Adds a regressor to the end of the current pipeline

        Parameters
        ----------
        regressor: Regressor
            The regressor to add to the pipeline

        Returns
        -------
        RegressorPipeline
            self

        """
        assert isinstance(regressor, Regressor), "Please provide a regressor object"
        return self.add_pipeline_element(RegressorPipelineElement(regressor))

    def train(self, instance: RegressionInstance):
        """train

        The train function of the Regressor. Calls pass_forward internally.

        Parameters
        ----------
        instance: RegressionInstance
            The instance to train on

        """
        self.pass_forward(instance)
        return self

    def predict(self, instance: Instance) -> TargetValue:
        """predict

        The predict function of the regressor.
        Calls pass_forward_predict internally and returns the prediction.

        Parameters
        ----------
        instance: Instance
            The instance to predict

        Returns
        -------
        TargetValue
            The prediction of the pipeline

        """
        instance, pred = self.pass_forward_predict(instance)
        return pred


class RandomSearchClassifierPE(ClassifierPipelineElement, Classifier):
    def __init__(
        self,
        classifier_class: Classifier,
        hyperparameter_ranges: dict,
        n_combinations: int,
        rng: np.random.Generator,
    ):
        # initialize the pipeline element but don't specify a learner
        super().__init__(learner=None)

        # assign the variables from the initializer
        self.classifier_class = classifier_class
        self.hyperparameter_ranges = hyperparameter_ranges
        self.n_combinations = n_combinations
        self.rng = rng

        # sample n_combinations of hyperparameters
        self.hyperparameters = []
        for _ in range(n_combinations):
            hp_combination = {
                hp_name: rng.choice(values)
                for hp_name, values in hyperparameter_ranges.items()
            }
            self.hyperparameters.append(hp_combination)

        # instantiate models
        self.models = [
            self.classifier_class(**hp_kwargs) for hp_kwargs in self.hyperparameters
        ]
        self.model_accuracy = [0.0 for _ in range(len(self.models))]
        self.seen_instances = 0

    def __str__(self):
        return f"RandomSearch({self.classifier_class.__name__!s})"

    def pass_forward(self, instance: LabeledInstance) -> Instance:
        """pass_forward

        Trains the learner on the provided instance; then returns the instance.

        Parameters
        ----------
        instance: Instance
            An instance to train the learner

        Returns
        -------
        Instance
            The instance that was provided to the function

        """
        # loop through all models, update their accuracy, and train them
        for model_idx, model in enumerate(self.models):
            y_hat = model.predict(instance)

            correct = int(y_hat == instance.y_index)
            old_acc = self.model_accuracy[model_idx]
            new_acc = (old_acc * self.seen_instances + correct) / (
                self.seen_instances + 1
            )
            self.model_accuracy[model_idx] = new_acc

            model.train(instance)
        self.seen_instances += 1
        return instance

    def pass_forward_predict(
        self, instance: Instance, prediction=None
    ) -> tuple[Instance, Any]:
        """pass_forward_predict

        Trains the learner on the provided instance; then returns the instance.

        Parameters
        ----------
        instance: Instance
            An instance to train the learner
        prediction: Any
            Most likely None, but could be anything in principle

        Returns
        -------
        Tuple[Instance, Any]
            The transformed instance and the prediction of the regressor

        """
        # find the best model, let it do the prediction
        best_model_idx = np.argmax(self.model_accuracy)
        best_model = self.models[best_model_idx]
        return instance, best_model.predict(instance)

    def train(self, instance: LabeledInstance):
        """train

        The train function of the Classifier. Calls pass_forward internally.

        Parameters
        ----------
        instance: LabeledInstance
            The instance to train on

        """
        self.pass_forward(instance)
        return self

    def predict(self, instance: Instance) -> LabelIndex | None:
        """predict

        The predict function of the classifier.
        Calls pass_forward_predict internally and returns the prediction.

        Parameters
        ----------
        instance: Instance
            The instance to predict

        Returns
        -------
        Optional[LabelIndex]
            The prediction of the pipeline

        """
        _inst, pred = self.pass_forward_predict(instance)
        return pred

    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        # TODO: Discuss how handle this
        raise NotImplementedError
