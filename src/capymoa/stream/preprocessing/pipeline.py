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
    """Basic building block for pipelines."""

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
        """Initialize the pipeline element with a classifier.

        :param learner: Classifier associated with this pipeline element.
        """
        self.learner = learner

    def pass_forward(self, instance: Instance) -> Instance:
        """Train the learner on an instance and return it.

        :param instance: Instance to train on.
        :returns: The input instance.
        """
        self.learner.train(instance)
        return instance

    def pass_forward_predict(
        self, instance: Instance, prediction: Any = None
    ) -> tuple[Instance, Any]:
        """Return the instance and the learner's prediction.

        The incoming ``prediction`` argument is ignored.

        :param instance: Instance to classify.
        :param prediction: Previous pipeline prediction, if any.
        :returns: The input instance and the classifier's prediction.
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
        """Initialize the pipeline element with a regressor.

        :param learner: Regressor associated with this pipeline element.
        """
        self.learner = learner

    def pass_forward(self, instance: Instance) -> Instance:
        """Train the learner on an instance and return it.

        :param instance: Instance to train on.
        :returns: The input instance.
        """
        self.learner.train(instance)
        return instance

    def pass_forward_predict(
        self, instance: Instance, prediction=None
    ) -> tuple[Instance, Any]:
        """Return the instance and the regressor's prediction.

        The incoming ``prediction`` argument is ignored.

        :param instance: Instance to use for prediction.
        :param prediction: Previous pipeline prediction, if any.
        :returns: The input instance and the regressor's prediction.
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
        """Initialize the pipeline element with a transformer.

        :param transformer: Transformer associated with this pipeline element.
        """
        self.transformer = transformer

    def pass_forward(self, instance: Instance) -> Instance:
        """Transform and return an instance.

        :param instance: Input instance.
        :returns: Transformed instance.
        """
        return self.transformer.transform_instance(instance)

    def pass_forward_predict(
        self, instance: Instance, prediction: Any = None
    ) -> tuple[Instance, Any]:
        """Transform the instance and pass the prediction through unchanged.

        :param instance: Input instance.
        :param prediction: Prediction to pass through.
        :returns: Transformed instance and unchanged prediction.
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
        """Initialize the pipeline element with a drift detector.

        The input preparation function must accept an instance and a prediction,
        for example ``prediction_is_correct(instance, pred)``. Its return value
        is passed to the drift detector.

        :param drift_detector: Drift detector to wrap.
        :param prepare_drift_detector_input_func: Function that prepares the
            value passed to the drift detector.
        """
        self.drift_detector = drift_detector
        self.prepare_drift_detector_input_func = prepare_drift_detector_input_func

    def pass_forward(self, instance: Instance) -> Instance:
        """Return the instance unchanged.

        The drift detector is updated by :meth:`pass_forward_predict`.

        :param instance: Input instance.
        :returns: The input instance.
        """
        return instance

    def pass_forward_predict(
        self, instance: Instance, prediction: Any = None
    ) -> tuple[Instance, Any]:
        """Update the detector and pass the instance and prediction through.

        The prediction may be ``None``, a classifier or regressor prediction,
        or another value accepted by
        ``prepare_drift_detector_input_func``.

        :param instance: Input instance.
        :param prediction: Prediction from earlier pipeline elements.
        :returns: The input instance and prediction.
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
        """Initialize the pipeline with its elements and input schema.

        :param pipeline_elements: Elements to add to the pipeline, in order.
        :param schema: Schema of input instances. If omitted, it is inferred
            from the first element that defines one.
        :param random_seed: Seed reported to satisfy the learner interface.
            The pipeline does not draw from it; its elements carry their own
            seeds.
        :param validate_schema: If ``True``, raise :class:`ValueError` when an
            added element expects a schema incompatible with the current output
            schema.
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
        """Append an element to the pipeline.

        :param element: Pipeline element to append.
        :returns: This pipeline.
        :raises ValueError: If the element's input schema is incompatible with
            the schema currently leaving the pipeline and schema validation is
            enabled.
        """
        self._check_schema_compatibility(element)
        self.elements.append(element)
        return self

    def add_transformer(self, transformer: Transformer):
        """Append a transformer to the pipeline.

        :param transformer: Transformer to append.
        :returns: This pipeline.
        """
        assert isinstance(transformer, Transformer), (
            "Please provide a Transformer object"
        )
        return self.add_pipeline_element(TransformerPipelineElement(transformer))

    def add_drift_detector(
        self,
        drift_detector: BaseDriftDetector,
        prepare_drift_detector_input_func: Callable,
    ):
        """Append a drift detector to the pipeline.

        .. note::
            This parameter was called ``get_drift_detector_input_func`` in
            earlier releases. It now matches the name
            :class:`DriftDetectorPipelineElement` has always used for the same
            argument because the mismatch between the two names was a trap.
            Passing the old keyword raises ``TypeError``; pass the argument
            positionally or use the new name.

        The input preparation function must accept an instance and a prediction,
        for example ``prediction_is_correct(instance, pred)``. Its return value
        is passed to the drift detector.

        :param drift_detector: Drift detector to append.
        :param prepare_drift_detector_input_func: Function that prepares the
            value passed to the drift detector.
        :returns: This pipeline.

        See Also
        --------
        capymoa.drift.monitors : Ready-made input functions.

        """
        assert isinstance(drift_detector, BaseDriftDetector)
        return self.add_pipeline_element(
            DriftDetectorPipelineElement(
                drift_detector, prepare_drift_detector_input_func
            )
        )

    def pass_forward(self, instance: Instance) -> Instance:
        """Pass an instance through all pipeline elements.

        Elements may transform the instance.

        :param instance: Instance to pass through the pipeline.
        :returns: Instance produced by the final pipeline element.
        """
        inst = instance
        for i, element in enumerate(self.elements):
            inst = element.pass_forward(inst)
        return inst

    def pass_forward_predict(
        self, instance: Instance, prediction: Any = None
    ) -> tuple[Instance, Any]:
        """Pass an instance and prediction through all pipeline elements.

        Use this to place a change detector after a pipeline that produces
        predictions. A base pipeline usually passes the prediction through
        unchanged.

        :param instance: Input instance.
        :param prediction: Prediction to pass through the pipeline.
        :returns: Instance and prediction produced by the final element.
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
        """Append a classifier to the pipeline.

        :param classifier: Classifier to append.
        :returns: This pipeline.
        """
        assert isinstance(classifier, Classifier), "Please provide a classifier object"
        return self.add_pipeline_element(ClassifierPipelineElement(classifier))

    def train(self, instance: LabeledInstance):
        """Train the pipeline on a labeled instance.

        :param instance: Labeled instance to train on.
        :returns: This pipeline.
        """
        self.pass_forward(instance)
        return self

    def predict(self, instance: Instance) -> LabelIndex | None:
        """Predict the class label for an instance.

        :param instance: Instance to classify.
        :returns: Predicted label, or ``None`` if unavailable.
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
        """Append a regressor to the pipeline.

        :param regressor: Regressor to append.
        :returns: This pipeline.
        """
        assert isinstance(regressor, Regressor), "Please provide a regressor object"
        return self.add_pipeline_element(RegressorPipelineElement(regressor))

    def train(self, instance: RegressionInstance):
        """Train the pipeline on a regression instance.

        :param instance: Regression instance to train on.
        :returns: This pipeline.
        """
        self.pass_forward(instance)
        return self

    def predict(self, instance: Instance) -> TargetValue:
        """Predict the target value for an instance.

        :param instance: Instance to use for prediction.
        :returns: Predicted target value.
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
        """Update model scores, train each model, and return the instance.

        :param instance: Labeled instance used to evaluate and train the models.
        :returns: The input instance.
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
        """Predict with the model that currently has the highest score.

        :param instance: Instance to classify.
        :param prediction: Previous pipeline prediction, if any; ignored.
        :returns: The input instance and the selected model's prediction.
        """
        # find the best model, let it do the prediction
        best_model_idx = np.argmax(self.model_accuracy)
        best_model = self.models[best_model_idx]
        return instance, best_model.predict(instance)

    def train(self, instance: LabeledInstance):
        """Train all candidate classifiers on a labeled instance.

        :param instance: Labeled instance to train on.
        :returns: This classifier.
        """
        self.pass_forward(instance)
        return self

    def predict(self, instance: Instance) -> LabelIndex | None:
        """Predict the class label using the highest-scoring classifier.

        :param instance: Instance to classify.
        :returns: Predicted label, or ``None`` if unavailable.
        """
        _inst, pred = self.pass_forward_predict(instance)
        return pred

    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        # TODO: Discuss how handle this
        raise NotImplementedError
