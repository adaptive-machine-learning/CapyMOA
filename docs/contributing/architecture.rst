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

..  warning::
    At the moment this parts of the documentation is aspirational rather than reflecting the current state of the project.

    Each domain shall implement its own::

        capymoa.{{domain}}                      # Domain-specific modules (listed above)
        - {{Domain}}Metrics          (class)    # Serializable metrics
        - evaluate_{{domain}}        (function) # Evaluation logic
        - *Algorithm                 (classes)  # Algorithm implementations

        capymoa.{{domain}}.base      (optional) # Abstract base classes for the module
        capymoa.{{domain}}.datasets  (optional) # Domain-specific datasets
        capymoa.{{domain}}.evaluate  (optional) # Public evaluation code
        capymoa.{{domain}}.plot      (optional) # Public plotting code

In addition to the research domain CapyMOA maintains some common features to simplify
implementation and facilitate interoperability:
