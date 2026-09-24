# Streaming classification

Learn how to classify data streams incrementally, one instance at a time.

```{toctree}
:maxdepth: 1

getting_started
evaluation
new_learner
parallel_ensembles
```

```{admonition} See also
:class: seealso

- {mod}`capymoa.classifier`
```

Classification assigns a discrete label to each instance as it arrives, updating the
model incrementally rather than retraining from scratch. These tutorials walk through
training and evaluating classifiers on data streams, covering the built-in learners,
evaluation procedures, and how to plug in your own models.
