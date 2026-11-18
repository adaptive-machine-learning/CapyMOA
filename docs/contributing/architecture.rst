Architecture
============

CapyMOA is organised into research domains.
Each research domain are maintained semi-independently by domain experts
(see ``CODEOWNERS``) with shared cross domain interoperability, continuous integration,
and documentation.
Domains can develop independently. This reflects how CapyMOA is built, allowing
researchers to own and contribute to their specific research domains.
Reduces the cognitive load required for a user to get started, while also driving the
discovery of related features within a domain.

``capymoa.anomaly``
    Streaming anomaly detection.

``capymoa.automl``
    Streaming automated machine learning.

``capymoa.classifier``
    Streaming classification.

``capymoa.cluster``
    Streaming clustering.

``capymoa.drift``
    Streaming concept and data drift detection.

``capymoa.feature``
    Streaming feature importance estimation.

``capymoa.ocl``
    Online (/streaming) continual learning.

``capymoa.regressor``
    Streaming regression.

``capymoa.ssl``
    Streaming semi-supervised learning

``capymoa.uncertainty``
    Streaming prediction intervals and uncertainty estimation.

Each domain implements its own::

    capymoa.{{domain}}                      # Domain-specific modules (listed above)
    - evaluate_{{domain}}        (function) # Evaluation logic
    - *Algorithm                 (classes)  # Algorithm implementations

    capymoa.{{domain}}.base      (optional) # Abstract base classes for the module
    capymoa.{{domain}}.datasets  (optional) # Domain-specific datasets
    capymoa.{{domain}}.evaluate  (optional) # Evaluators and result types (TypedDict)
    capymoa.{{domain}}.plot      (optional) # Public plotting code

``evaluate_{{domain}}`` evaluates one learner. Domains that can compare learners
on one pass over a stream (classifier, regressor, anomaly and uncertainty) also
take a mapping of names to learners and return a dict of results by name.
``{{Domain}}Results`` is a plain ``TypedDict`` with a fixed set of keys, so
``pandas.DataFrame([r1, r2])`` is a tidy table. It lives in
``capymoa.{{domain}}.evaluate``, next to the evaluator classes (such as
``ClassificationEvaluator``) for use in custom loops. ``evaluate_{{domain}}`` is
in ``capymoa/{{domain}}/_evaluate.py`` and exported from the domain.

The domains that follow this layout are ``capymoa.classifier``,
``capymoa.regressor``, ``capymoa.uncertainty`` (prediction intervals),
``capymoa.ssl``, ``capymoa.anomaly`` and ``capymoa.ocl``.

``capymoa.evaluation`` holds only the parts shared by all domains:

- ``RunInfo``, the base of every result (learner, stream, timing, windowed
  table, stored targets and predictions and drifts).
- ``plot_windowed_results`` in ``capymoa.evaluation.plot``.
- ``prequential_evaluation``, which calls the ``evaluate_{{domain}}`` function that
  matches the type of the learner.

Seeds & Determinism
-------------------

All CapyMOA classes that use random number generators, including those that delegate random number generators to dependencies, shall be seeded using a ``random_seed: int | None = None`` parameter in the constructor.

Except for datasets, the default ``random_seed`` shall be ``None``, indicating that it is randomly generated at runtime from NumPy's global random number generator.

As a consequence, learners without an explicitly set seed will produce non-deterministic results.

CapyMOA classes may provide a ``random_seed`` attribute to indicate the seed used for reproducibility.

Results with the same seed will be identical on the same platform and with the same dependency environment.

