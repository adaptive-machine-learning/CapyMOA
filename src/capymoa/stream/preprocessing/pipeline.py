from __future__ import annotations

from abc import abstractmethod
from typing import Optional, List, Protocol, Tuple, Any, Callable

import numpy as np

from capymoa.base import Classifier, Regressor
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.instance import LabeledInstance, Instance, RegressionInstance

from capymoa.stream.preprocessing.transformer import Transformer
from capymoa.type_alias import LabelProbabilities, LabelIndex, TargetValue


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
    ) -> Tuple[Instance, Any]:
        raise NotImplementedError

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
    ) -> Tuple[Instance, Any]:
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

    def __str__(self):
        return "PE({})".format(str(self.learner))


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
    ) -> Tuple[Instance, Any]:
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

    def __str__(self):
        return "PE({})".format(str(self.learner))


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
    ) -> Tuple[Instance, Any]:
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

    def __str__(self):
        return "PE({})".format(str(self.transformer))


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
    ) -> Tuple[Instance, Any]:
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
        return "PE({})".format(str(self.drift_detector))


class BasePipeline(PipelineElement):
    """
    The base class for other types of pipelines. Supports transformers and drift detectors.
    """

    def __init__(self, pipeline_elements: List[PipelineElement] | None = None):
        """__init__

        Initializes the base pipeline with a list of pipeline elements.

        Parameters
        ----------
        pipeline_elements: List[PipelineElement]
            The elements the pipeline consists of

        """
        self.elements: List[PipelineElement] = (
            [] if pipeline_elements is None else pipeline_elements
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

        """
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
        self.elements.append(TransformerPipelineElement(transformer))
        return self

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
        self.elements.append(
            DriftDetectorPipelineElement(drift_detector, get_drift_detector_input_func)
        )
        return self

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
    ) -> Tuple[Instance, Any]:
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
        s = ""
        for i, element in enumerate(self.elements):
            s += str(element)
            s += " | "
        return s


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
        self.elements.append(ClassifierPipelineElement(classifier))
        return self

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

    def predict(self, instance: Instance) -> Optional[LabelIndex]:
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
        inst, pred = self.pass_forward_predict(instance)
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
        self.elements.append(RegressorPipelineElement(regressor))
        return self

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
        super(RandomSearchClassifierPE, self).__init__(learner=None)

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
        return f"RandomSearch({str(self.classifier_class.__name__)})"

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
    ) -> Tuple[Instance, Any]:
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

    def predict(self, instance: Instance) -> Optional[LabelIndex]:
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
        inst, pred = self.pass_forward_predict(instance)
        return pred

    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        # TODO: Discuss how handle this
        raise NotImplementedError
