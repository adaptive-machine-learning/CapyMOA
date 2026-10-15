Release Notes
=============

Version 0.16
------------

Results are now typed dictionaries. Every ``evaluate_*`` function returns a
plain :class:`dict`. Each domain has its own result type, for example
:class:`~capymoa.classifier.evaluate.ClassifierResults`. Read a metric with a
key, such as ``results["cumulative"]["accuracy"]``, instead of a method.

Breaking changes
~~~~~~~~~~~~~~~~

* **Positional arguments.** :func:`~capymoa.evaluation.prequential_evaluation`
  takes ``stream`` and ``learner`` by position. All other arguments must be
  keywords. Calls such as ``prequential_evaluation(stream, learner, 1000)``
  now raise ``TypeError``. Use ``max_instances=1000``.
* **Results are dicts.** The old ``PrequentialResults`` object is gone. Use
  keys and item access, not methods.
* **Windowed per-class metrics moved.** Columns such as ``precision_0`` are no
  longer in the windowed results. ``results["windowed_per_class"]`` holds them
  instead, with one row per window and class. ``pd.DataFrame`` turns it into a
  long table with ``instances``, ``label``, ``precision``, ``recall`` and
  ``f1_score`` columns.
* **Whole-stream metrics moved** under ``results["cumulative"]``, for example
  ``results["cumulative"]["accuracy"]``. Per-class metrics are in
  ``results["cumulative_per_class"]``, like ``windowed_per_class``.
* **Plot helpers moved** to the domain modules, see the table below.
  ``write_to_file`` is gone. Results are plain dicts, so save them with
  :mod:`pickle`.

New metrics
~~~~~~~~~~~

* Classifiers and SSL learners: ``precision_macro``, ``recall_macro`` and
  ``f1_score_macro``, the mean over the classes. Also ``windowed_per_class``.
* Regressors: ``rrmse``.
* Anomaly detectors: ``accuracy``, ``kappa``, ``kappa_m``, ``recall``,
  ``g_mean`` and ``pos_neg_ratio``. The windowed results also have
  ``periodical_auc``.
* SSL with the Java loop: ``pseudo_label_accuracy``. The Python loop does not
  make pseudo labels, so this key is absent there.
* :func:`~capymoa.uncertainty.evaluate_prediction_interval` and
  :func:`~capymoa.anomaly.evaluate_anomaly` accept ``optimise`` and
  ``batch_size``. Before, ``prequential_evaluation`` raised ``TypeError`` for
  these arguments on those learners.

Migration guide
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Before (0.15 and earlier)
     - Now (0.16)
   * - ``prequential_evaluation(stream, learner, 1000)``
     - ``prequential_evaluation(stream, learner, max_instances=1000)``
   * - ``prequential_ssl_evaluation(stream, learner, ...)``
     - :func:`capymoa.ssl.evaluate_ssl` (same arguments)
   * - ``prequential_evaluation_anomaly(stream, learner, ...)``
     - :func:`capymoa.anomaly.evaluate_anomaly` (same arguments)
   * - ``prequential_evaluation_multiple_learners(stream, learners)``
     - :func:`capymoa.classifier.evaluate_classifier` with a mapping of
       learners. Use :func:`capymoa.regressor.evaluate_regressor` for
       regressors.
   * - ``ocl_train_eval_loop(learner, train, test, ...)``
     - :func:`capymoa.ocl.evaluate_ocl` (same arguments). It returns an
       ``OCLResults`` dict, not ``OCLMetrics``.
   * - ``results.cumulative.accuracy()``
     - ``results["cumulative"]["accuracy"]``
   * - ``results.windowed.accuracy()``
     - ``results["windowed"]["accuracy"]``
   * - ``results.wallclock()``, ``results.cpu_time()``
     - ``results["wallclock"]``, ``results["cpu_time"]``
   * - ``results.ground_truth_y()``
     - ``results["y_true"]``. Set ``store_y=True``.
   * - ``results.predictions()``
     - ``results["y_pred"]``. Set ``store_predictions=True``.
   * - ``results.other_metrics()``
     - The keys themselves, for example ``results["unlabeled"]`` for SSL.
   * - ``results.metrics_per_window()``
     - ``pd.DataFrame(results["windowed"])``. Per-class columns such as
       ``precision_0`` are in ``pd.DataFrame(results["windowed_per_class"])``.
   * - ``results.write_to_file(path)``
     - ``pickle.dump(results, file)``. Results are plain dicts.
   * - ``capymoa.evaluation.ClassificationEvaluator``, ``ClassificationWindowedEvaluator``
     - :mod:`capymoa.classifier.evaluate`
   * - ``capymoa.evaluation.RegressionEvaluator``, ``RegressionWindowedEvaluator``
     - :mod:`capymoa.regressor.evaluate`
   * - ``capymoa.evaluation.PredictionIntervalEvaluator``, ``PredictionIntervalWindowedEvaluator``
     - :mod:`capymoa.uncertainty.evaluate`
   * - ``capymoa.evaluation.AnomalyDetectionEvaluator``
     - :mod:`capymoa.anomaly.evaluate`
   * - ``capymoa.evaluation.ClusteringEvaluator``
     - :mod:`capymoa.cluster.evaluate`
   * - ``capymoa.evaluation.visualization.plot_windowed_results``
     - :func:`capymoa.evaluation.plot.plot_windowed_results`
   * - ``capymoa.evaluation.visualization.plot_regression_results``, ``plot_predictions_vs_ground_truth``
     - :mod:`capymoa.regressor.plot`
   * - ``capymoa.evaluation.visualization.plot_prediction_interval``
     - :mod:`capymoa.uncertainty.plot`
   * - ``capymoa.evaluation.visualization.plot_clustering_state``, ``plot_clustering_evolution``
     - :mod:`capymoa.cluster.plot`
