# Choose a backend

A new method can be written in one of four ways. This page helps you pick one,
and says what can be mixed.

| Backend | Use it when | Cost |
|---|---|---|
| Java (MOA) | The method exists in MOA, or speed matters most. | Needs the Java build (see [](moa/moa_integration.md)). |
| Pure Python | The method is short, or uses NumPy only. | Slower than Java. |
| scikit-learn | An online (`partial_fit`) estimator already exists. | Slow per instance. |
| PyTorch | The method is a neural network. | Needs the `torch` extra. |

## How to decide

1. **Is it in MOA already?** Wrap it. Do not rewrite it.
2. **Will it be an ensemble member or run in a long stream?** Prefer Java or
   fast Python. Each Python call from Java is costly.
3. **Is it a neural network?** Use PyTorch.
4. **Otherwise** write pure Python. Move it to Java later if profiling says so.

## What can be mixed

A MOA ensemble can hold a Python member. A CapyMOA `Classifier`, `Regressor` or
drift detector written in Python (or backed by scikit-learn) can be set on a MOA
class option. The public ensembles do not use this yet. It is the base for adding
Python members to them later.

Limits:

* **One thread.** Python members do not work with the multi-threaded
  (`number_of_jobs > 1`) MOA ensembles.
* **Interfaces only.** The MOA option must take an interface
  (`Classifier`, `Regressor` or `ChangeDetector`). Options that need a class
  (such as `ARFHoeffdingTree` in the random forests) only take MOA objects.
* **Java copies.** Some ensembles (such as gradient boosting) copy members with
  Java serialisation. A Python member can not do this.
* **Speed.** On `Electricity` (10 000 instances, 10 members), a MOA
  `HoeffdingTree` ensemble ran at about 41 000 inst/s. With a Python
  `PassiveAggressiveClassifier` it ran at about 380 inst/s. The Python learner
  itself is the main cost, not the call from Java.
* **Copies are fresh.** MOA copies the base learner to make each member. A
  Python copy is a new untrained learner with the same parameters.

## How it works

MOA builds ensemble members from a CLI string. A Python object has none.
Instead, `capymoa.core.moa._python_proxy` makes a JPype proxy that implements the
MOA `Classifier`, `Regressor` or `ChangeDetector` interface and calls the Python
object. The proxy is set as the value of the MOA option, which MOA accepts in
place of a string.

To use it in an ensemble, call `resolve_base_learner` (or `resolve_base_regressor`,
`resolve_drift_detector`). Each returns a CLI string and a `java_options` map. Pass
the map to `MOAClassifier` or `MOARegressor`, which sets the options before the
learner is prepared.
