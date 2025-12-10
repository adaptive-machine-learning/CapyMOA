import random
from capymoa.base import (
    Classifier,
)
from typing import Dict, Any
from capymoa.stream import Schema
from capymoa.evaluation import ClassificationEvaluator
from capymoa.automl._utils import (
    generate_parameter_combinations,
    create_capymoa_classifier,
)
import json


class EpsilonGreedy:
    """Epsilon-Greedy bandit policy for model selection.

    This policy selects the best model with probability ``1 - epsilon`` and explores
    other models with probability ``epsilon``. During the burn-in period, it always
    explores to gather initial information about all models.

    >>> from capymoa.automl import EpsilonGreedy
    >>> policy = EpsilonGreedy(epsilon=0.1, burn_in=50)
    >>> policy.epsilon
    0.1

    .. seealso::

        :class:`~capymoa.automl.BanditClassifier`
    """

    def __init__(self, epsilon: float = 0.1, burn_in: int = 100):
        """Construct a new Epsilon-Greedy policy.

        :param epsilon: Probability of exploring a random model (default: ``0.1``).
        :param burn_in: Number of initial rounds dedicated to exploration (default: ``100``).
        """
        self.epsilon = epsilon
        """Probability of exploring a random model."""

        self.burn_in = burn_in
        """Number of initial rounds where all models are explored to collect initial statistics."""

        self.n_arms = 0
        """Number of available models (arms)."""

        self.arm_rewards = []
        """Cumulative reward values for each model (arm)."""

        self.arm_counts = []
        """Number of times each arm has been pulled."""

        self.total_pulls = 0
        """Total number of model selections performed."""

    def initialize(self, n_arms):
        """Initialize the policy with a given number of arms."""
        self.n_arms = n_arms
        self.arm_rewards = [0.0] * n_arms
        self.arm_counts = [0] * n_arms
        self.total_pulls = 0

    def get_best_arm_idx(self, available_arms):
        best_arm = max(
            available_arms,
            key=lambda arm: self.arm_rewards[arm] / max(1, self.arm_counts[arm]),
        )
        return best_arm

    def pull(self, available_arms):
        """Select which arms to pull based on the epsilon-greedy policy."""
        if self.total_pulls < self.burn_in:
            # During burn-in, explore all available arms
            return available_arms

        # With probability epsilon, explore a random arm
        if random.random() < self.epsilon:
            return [random.choice(available_arms)]

        # Otherwise, exploit the best arm
        best_arm = max(
            available_arms,
            key=lambda arm: self.arm_rewards[arm] / max(1, self.arm_counts[arm]),
        )
        return [best_arm]

    def update(self, arm, reward):
        """Update the policy with the observed reward for the pulled arm."""
        self.arm_rewards[arm] += reward
        self.arm_counts[arm] += 1
        self.total_pulls += 1

    def get_arm_stats(self):
        """Get statistics about each arm's performance."""
        return {
            "rewards": self.arm_rewards.copy(),
            "counts": self.arm_counts.copy(),
            "means": [r / max(1, c) for r, c in zip(self.arm_rewards, self.arm_counts)],
        }


class BanditClassifier(Classifier):
    """Bandit-based model selection for streaming classification.

    Each base classifier is associated with an arm of a multi-armed bandit.
    At each training step, the bandit policy selects which model to train
    (i.e., which arm to pull). The reward corresponds to the model's
    performance on the current instance. The best-performing model is then
    used for prediction [#robbins1952]_.

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import HoeffdingTree
    >>> from capymoa.automl import BanditClassifier, EpsilonGreedy
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = BanditClassifier(
    ...     schema=schema,
    ...     base_classifiers=[HoeffdingTree],
    ...     policy=EpsilonGreedy(epsilon=0.1, burn_in=100)
    ... )
    >>> instance = next(stream)
    >>> learner.train(instance)

    .. seealso::

        :class:`~capymoa.automl.EpsilonGreedy`

    .. [#robbins1952] Robbins, H. (1952). *Some aspects of the sequential design of experiments.*
       Bulletin of the American Mathematical Society, 58(5), 527–535.
    """

    def __init__(
        self,
        schema: Schema = None,
        random_seed: int = 1,
        base_classifiers: list = None,
        config_file: str = None,
        metric: str = "accuracy",
        policy: EpsilonGreedy = None,
        verbose: bool = False,
    ):
        """Construct a Bandit-based model selector.

        :param schema: The schema of the stream.
        :param random_seed: Random seed used for initialization.
        :param base_classifiers: List of base classifier classes to consider.
        :param config_file: Path to a JSON configuration file with model hyperparameters.
        :param metric: The metric used to evaluate model performance. Defaults to ``"accuracy"``.
        :param policy: The bandit policy used to choose which model to train (e.g., :class:`~capymoa.automl.EpsilonGreedy`).
        :param verbose: If True, print progress information during training.
        """
        super().__init__(schema=schema, random_seed=random_seed)

        self.config_file = config_file
        self.base_classifiers = base_classifiers
        self.metric = metric
        self.policy = policy
        self.verbose = verbose
        self.log_cnt = 0
        self.log_point = 5000

        # Initialize policy if not provided
        if self.policy is None:
            self.policy = EpsilonGreedy(epsilon=0.1, burn_in=100)

        # Initialize models based on configuration
        self._initialize_models()

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

        # Initialize policy with number of arms
        self.policy.initialize(len(self.active_models))

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
        """Train the selected model(s) on the given instance."""
        # Get the arm(s) to pull from the policy
        arm_ids = self.policy.pull(range(len(self.active_models)))

        # Train and evaluate each selected model
        for arm_id in arm_ids:
            model = self.active_models[arm_id]
            metric = self.metrics[arm_id]

            # Make prediction for evaluation before training
            y_pred = model.predict(instance)
            # Update metric with prediction
            metric.update(instance.y_index, y_pred)
            # Train the model
            model.train(instance)

            # Update the policy with the reward (metric value)
            reward = metric.accuracy() if self.metric == "accuracy" else metric.get()
            self.policy.update(arm_id, reward)

            # Check if this model is better than our current best
            if metric.accuracy() > self.metrics[self._best_model_idx].accuracy():
                self._best_model_idx = arm_id

            # Add verbose logging
        self.log_cnt += 1
        if self.verbose and self.log_cnt >= self.log_point:
            self.log_cnt = 0
            current_accuracy = metric.accuracy()
            model_performances = [(i, self.metrics[i].accuracy()) for i in arm_ids]
            top_models = sorted(model_performances, key=lambda x: x[1], reverse=True)[
                :3
            ]

            print(f"\nChosen model: {model}")
            print(f"Current accuracy: {current_accuracy:.4f}")

            # Print top 3 models if there are many models
            if len(model_performances) >= 3:
                print("\nTop models:")
                for i, (model_idx, acc) in enumerate(top_models):
                    model_name = str(self.active_models[model_idx])
                    print(f"  {i + 1}. {model_name} - Accuracy: {acc:.4f}")

    def predict(self, instance):
        """Predict the class label for the given instance using the best model."""
        if not self.active_models:
            raise ValueError(
                "No active models available. Please train the classifier first."
            )
        idx = self.policy.get_best_arm_idx(range(len(self.active_models)))
        # Use the best performing model for predictions
        # idx = self._best_model_idx
        return self.active_models[idx].predict(instance)

    def predict_proba(self, instance):
        """Predict class probabilities for the given instance using the best model."""
        if not self.active_models:
            raise ValueError(
                "No active models available. Please train the classifier first."
            )

        idx = self.policy.get_best_arm_idx(range(len(self.active_models)))
        # Use the best performing model for predictions
        # idx = self._best_model_idx
        return self.active_models[idx].predict_proba(instance)

    def __str__(self):
        """Return a string representation of the model."""
        return "BanditClassifier"

    @property
    def best_model(self):
        """Return the current best model."""
        idx = self.policy.get_best_arm_idx(range(len(self.active_models)))
        return self.active_models[idx]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current state of the classifier.

        Returns:
            Dictionary containing classifier information
        """
        # Get performance metrics for all models
        model_performances = {
            str(self.active_models[i]): self.metrics[i].accuracy()
            for i in range(len(self.active_models))
        }
        sorted_dict = dict(
            sorted(model_performances.items(), key=lambda item: item[1], reverse=True)
        )
        # Get top-performing models
        top_models = []
        max_models = min(5, len(self.active_models))
        i = 0
        # idx = self.policy.get_best_arm_idx(range(len(self.active_models)))
        for key, value in sorted_dict.items():
            if i >= max_models:
                break
            top_models.append({"model": key, "accuracy": value})
            i += 1

        return {
            "total_models": len(self.active_models),
            "best_model_index": self._best_model_idx,
            "model_performances": sorted_dict,
            "best_model_accuracy": (
                self.metrics[self._best_model_idx].accuracy() if self.metrics else None
            ),
            "top_models": top_models,
        }
