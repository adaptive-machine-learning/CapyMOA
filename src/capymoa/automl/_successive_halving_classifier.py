from capymoa.base import (
    Classifier,
)
from typing import Dict, Any
from capymoa.stream import Schema
import math
import json
from capymoa.evaluation import ClassificationEvaluator
from capymoa.automl._utils import (
    generate_parameter_combinations,
    create_capymoa_classifier,
)


class SuccessiveHalvingClassifier(Classifier):
    """Successive Halving Classifier for model selection in streaming scenarios.

    This method applies the principle of *Successive Halving* [#jamieson2016]_
    to incrementally allocate more computational budget to the most promising
    models while discarding poorly performing ones. It progressively narrows
    the pool of candidates as more data is observed, improving efficiency in
    streaming model selection.

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import HoeffdingTree
    >>> from capymoa.automl import SuccessiveHalvingClassifier
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = SuccessiveHalvingClassifier(
    ...     schema=schema,
    ...     base_classifiers=[HoeffdingTree],
    ...     eta=2.0,
    ...     evaluation_metric="accuracy"
    ... )
    >>> instance = next(stream)
    >>> learner.train(instance)

    .. seealso::

        :class:`~capymoa.automl.BanditClassifier`

    .. [#jamieson2016] Jamieson, K. & Talwalkar, A. (2016, May).
       *Non-stochastic best arm identification and hyperparameter optimization.*
       In Artificial Intelligence and Statistics (pp. 240–248). PMLR.
       https://proceedings.mlr.press/v51/jamieson16.html
    """

    def __init__(
        self,
        schema: Schema | None = None,
        random_seed: int = 1,
        base_classifiers: list[type[Classifier]] | None = None,
        config_file: str | None = None,
        max_instances: int | None = None,
        budget: int | None = None,
        eta: float = 2.0,
        min_models: int = 1,
        evaluation_metric: str = "accuracy",
        verbose: bool = False,
    ):
        """Construct a Successive Halving Classifier.

        :param schema: The schema of the stream.
        :param random_seed: Random seed used for reproducibility.
        :param base_classifiers: List of base classifier classes to consider.
        :param config_file: Path to a JSON configuration file with model hyperparameters.
        :param max_instances: Maximum number of instances to process per model.
            If provided, the total budget is computed as ``2 * n_models * max_instances / eta``.
        :param budget: Total training budget (number of instances across all models).
            Ignored if ``max_instances`` is provided.
        :param eta: Reduction factor controlling how many models advance to the next round.
        :param min_models: Minimum number of models to maintain in successive rounds.
        :param evaluation_metric: Metric used to evaluate models. Defaults to ``"accuracy"``.
        :param verbose: Whether to print progress information during training.
        """
        super().__init__(schema=schema, random_seed=random_seed)

        self.config_file = config_file
        self.base_classifiers = base_classifiers
        self.max_instances = max_instances
        self.budget = budget
        self.eta = eta
        self.min_models = min_models
        self.evaluation_metric = evaluation_metric
        self.verbose = verbose

        # Initialize models based on configuration
        self._initialize_models()
        # Calculate budget if max_instances is provided
        if self.max_instances is not None:
            self.budget = math.ceil((2 * self._n * self.max_instances) / self.eta)
            if self.verbose:
                print(f"Budget automatically calculated: {self.budget} total updates")
                print(
                    f"Formula: (2 * {self._n} models * {self.max_instances} instances) / {self.eta}"
                )
        elif self.budget is None:
            # Default budget if neither max_instances nor budget is provided
            self.budget = 10000
            if self.verbose:
                print(f"Using default budget: {self.budget}")

        # Recalculate resource allocation based on final budget
        self._r = math.floor(
            self.budget / (self._s * math.ceil(math.log(max(2, self._n), self.eta)))
        )

        # Track the best model
        self._best_model_idx = 0

    def _initialize_models(self):
        """Initialize models based on configuration."""
        # Validate that we have at least one source of models
        if self.base_classifiers is None and self.config_file is None:
            raise ValueError("Either base_classifiers or config_file must be provided")

        # Initialize state variables
        self.active_models = []  # List of active model instances
        self.metrics = []  # List of evaluation metrics for each model

        # If using a config file, load and process it
        if self.config_file is not None:
            if self.verbose:
                print(f"Loading model configurations from {self.config_file}")
            self._load_model_configurations()
        else:
            # Use the provided base classifiers directly
            if self.verbose:
                print(f"Using {len(self.base_classifiers)} provided base classifiers")
            for model in self.base_classifiers:
                # Check if model is already instantiated or is a class
                if isinstance(model, Classifier):
                    # Model is already instantiated, use it directly
                    self.active_models.append(model)
                else:
                    # Model is a class, instantiate it
                    clf_instance = model(schema=self.schema)
                    self.active_models.append(clf_instance)

                # Create an evaluator for this model
                self.metrics.append(ClassificationEvaluator(schema=self.schema))

        # Update state variables based on the total number of models
        self._n = len(self.active_models)  # Total number of models
        self._s = self._n  # Current number of active models
        self._n_rungs = 0  # Number of completed rungs
        self._iterations = 0  # Number of instances processed in current rung
        self._budget_used = 0  # Total budget used so far
        self._rankings = list(range(self._n))  # Current model rankings

    def _load_model_configurations(self):
        """Load model configurations from a JSON file."""
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)

            # Process algorithms section of the config
            algorithms = config.get("algorithms", [])

            # If there are no algorithms defined, raise an error
            if not algorithms:
                raise ValueError("No algorithms defined in the configuration file")

            # Process each algorithm and its parameter configurations
            for algo_config in algorithms:
                algorithm_name = algo_config.get("algorithm")
                parameters = algo_config.get("parameters", [])

                # Generate all parameter combinations
                param_combinations = generate_parameter_combinations(parameters)

                # Create a classifier for each parameter combination
                for params in param_combinations:
                    try:
                        # Create classifier instance
                        clf = create_capymoa_classifier(
                            algorithm_name, params, self.schema
                        )

                        if clf is not None:
                            self.active_models.append(clf)
                            self.metrics.append(
                                ClassificationEvaluator(schema=self.schema)
                            )

                            if self.verbose:
                                param_str = ", ".join(
                                    [f"{p['parameter']}={p['value']}" for p in params]
                                )
                                print(
                                    f"Added model: {algorithm_name} with parameters: {param_str}"
                                )
                    except Exception as e:
                        print(
                            f"Warning: Failed to create model {algorithm_name} with parameters {params}: {str(e)}"
                        )

        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Error loading configuration file: {str(e)}")

    def train(self, instance):
        # Train only active models
        for i in self._rankings[: self._s]:
            model = self.active_models[i]
            metric = self.metrics[i]

            # Make prediction for evaluation before training
            y_pred = model.predict(instance)
            # Update metric with prediction
            metric.update(instance.y_index, y_pred)
            # Train the model
            model.train(instance)

            # Check if this model is better than our current best
            if metric.accuracy() > self.metrics[self._best_model_idx].accuracy():
                self._best_model_idx = i

        # Increment iteration counter
        self._iterations += 1

        # Check if we've completed a rung and have more than one model
        if self._s > self.min_models and self._iterations >= self._r:
            # Increment rung counter
            self._n_rungs += 1
            # Update budget used
            self._budget_used += self._s * self._r

            # Rank models by their performance
            self._rankings[: self._s] = sorted(
                self._rankings[: self._s],
                key=lambda i: self.metrics[i].accuracy(),
                reverse=True,  # Higher accuracy is better
            )

            # self._best_model_idx = self._rankings[0]

            # Determine how many models to keep
            cutoff = max(self.min_models, math.ceil(self._s / self.eta))

            # Print status if verbose
            if self.verbose:
                best_score = self.metrics[self._rankings[0]].accuracy()
                print(
                    "\t".join(
                        (
                            f"[Rung {self._n_rungs}]",
                            f"{self._s - cutoff} models removed",
                            f"{cutoff} models left",
                            f"{self._r} instances per model",
                            f"budget used: {self._budget_used}",
                            f"budget left: {self.budget - self._budget_used}",
                            f"best accuracy: {best_score:.4f}",
                        )
                    )
                )

                # Show the top 3 models if there are many
                if self._s > 3:
                    print("Top models:")
                    for i, idx in enumerate(self._rankings[: min(3, cutoff)]):
                        model = self.active_models[idx]
                        accuracy = self.metrics[idx].accuracy()
                        print(f"  {i + 1}. {model} - Accuracy: {accuracy:.4f}")

            # Update the number of active models
            self._s = cutoff

            # Recalculate the number of instances for the next rung
            remaining_rungs = max(
                1, math.ceil(math.log(self._n, self.eta)) - self._n_rungs
            )
            self._r = math.floor(
                (self.budget - self._budget_used) / (self._s * remaining_rungs)
            )

            # Reset iteration counter for the next rung
            self._iterations = 0

    def predict(self, instance):
        if not self.active_models:
            raise ValueError(
                "No active models available. Please train the classifier first."
            )

        # Use the best performing model for predictions
        return self.active_models[self._best_model_idx].predict(instance)

    def predict_proba(self, instance):
        if not self.active_models:
            raise ValueError(
                "No active models available. Please train the classifier first."
            )

        # Use the best performing model for predictions
        return self.active_models[self._best_model_idx].predict_proba(instance)

    def __str__(self):
        """Return a string representation of the model."""
        return "SuccessiveHalvingClassifier"

    @property
    def best_model(self):
        """Return the current best model."""
        return self.active_models[self._best_model_idx]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current state of the classifier.

        Returns:
            Dictionary containing classifier information
        """
        # Calculate active models based on rankings
        # active_models = [self.active_models[i] for i in self._rankings[:self._s]]
        # Get performance metrics for all models
        model_performances = {
            str(self.active_models[i]): self.metrics[i].accuracy()
            for i in range(len(self.active_models))
        }

        # Get top-performing models
        top_models = []
        for i in range(min(5, self._s)):
            if i < len(self._rankings):
                idx = self._rankings[i]
                top_models.append(
                    {
                        "model": str(self.active_models[idx]),
                        "accuracy": self.metrics[idx].accuracy(),
                    }
                )

        return {
            "active_models": self._s,
            "total_models": self._n,
            "current_rung": self._n_rungs,
            "max_instances": self.max_instances,
            "total_budget": self.budget,
            "budget_used": self._budget_used,
            "budget_left": self.budget - self._budget_used,
            "iterations_in_current_rung": self._iterations,
            "best_model_index": self._best_model_idx,
            "model_performances": model_performances,
            "best_model_accuracy": (
                self.metrics[self._best_model_idx].accuracy() if self.metrics else None
            ),
            "top_models": top_models,
        }
