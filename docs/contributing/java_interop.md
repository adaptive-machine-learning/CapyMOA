# Implement a Method in Java, Use It From Python

CapyMOA wraps the Java library [MOA](https://moa.cms.waikato.ac.nz/) using
[JPype](https://jpype.readthedocs.io/). If you're adding a new learner
backed by a MOA Java class, you'll usually want to write and test the Java
side first, before writing the permanent Python wrapper described in the
[FAQ](faq.md#what-does-a-learner-implement).

This guide walks through getting a Java class onto the classpath so you can
call it from Python, using a small worked example: `AlwaysPositive`, a
trivial `moa.classifiers.Classifier` that always predicts class index 0.
It's compiled against real MOA interfaces
(`moa.classifiers.AbstractClassifier`,
`com.yahoo.labs.samoa.instances.Instance`), so the compile and classpath
steps below are the ones you'll actually hit with a real classifier. See
MOA's own [Introduction to the API of
MOA](https://moa.cms.waikato.ac.nz/tutorial-2-introduction-to-the-api-of-moa/)
tutorial for a full walkthrough of `AbstractClassifier` and the methods
implemented below (`trainOnInstanceImpl`, `getVotesForInstance`, and so on).

```java
package org.capymoa.example;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.Measurement;

public class AlwaysPositive extends AbstractClassifier implements Classifier {

    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "A trivial classifier that always predicts class index 0.";
    }

    @Override
    public void resetLearningImpl() {
        // No state to reset.
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        // Never updates.
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        double[] votes = new double[inst.numClasses()];
        votes[0] = 1.0;
        return votes;
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        out.append("AlwaysPositive: predicts class 0 for every instance.");
    }
}
```

There are two ways to get a class like this onto CapyMOA's classpath: build
all of MOA, or add just this one class via JPype. Pick based on how big
your Java change is.

## Background: how CapyMOA loads MOA

`import capymoa` starts the JVM as the very first thing it does
(`src/capymoa/__init__.py`), before any other submodule is imported. That
startup, in `src/capymoa/_prepare_jpype.py::_start_jpype()`, does three
things:

1. Resolves the MOA jar via `capymoa_moa_jar()`
   (`src/capymoa/env.py`) — the bundled `src/capymoa/jar/moa.jar` by
   default, or the `CAPYMOA_MOA_JAR` environment variable if it's set.
2. Adds that jar to the classpath with `jpype.addClassPath(moa_jar)`.
3. Starts the JVM with `jpype.startJVM(...)`.

Crucially, it does none of this if `jpype.isJVMStarted()` is already
`True` — it just returns. That's the hook the second approach below uses.

## Approach 1: Build the Whole MOA Project

Use this when your Java change spans multiple classes, touches MOA's own
build, or you want a fully faithful test that matches what a real release
would do.

1. Add `AlwaysPositive.java` to your MOA source checkout, following its
   existing package layout (for example, alongside
   `moa.classifiers.functions.MajorityClass`, which CapyMOA already wraps
   in `src/capymoa/classifier/_majority_class.py`).
2. Build MOA with its own build tooling to produce a jar.
3. Point CapyMOA at your built jar instead of the bundled one:

   ```bash
   export CAPYMOA_MOA_JAR=/path/to/your/moa/build/output/moa.jar
   python -c "import capymoa; capymoa.about()"
   ```

   `capymoa.about()` prints the resolved `CAPYMOA_MOA_JAR` path and a
   hash of the jar, so you can confirm CapyMOA picked up your build:

   ```console
   $ python -c "import capymoa; capymoa.about()"
   CapyMOA 0.14.0
     CAPYMOA_DATASETS_DIR: data
     CAPYMOA_MOA_JAR:      /path/to/your/moa/build/output/moa.jar
     CAPYMOA_JVM_ARGS:     ['-Xmx8g', '-Xss10M']
     JAVA_HOME:            /usr/lib/jvm/java-21-openjdk-amd64
     MOA version:          <hash of your build>
     JAVA version:         21.0.11
   ```

Remember to `unset CAPYMOA_MOA_JAR` (or open a new shell) once you're done,
so CapyMOA goes back to using the bundled jar. See the [Update MOA
guide](update_moa.md), which relies on the same variable and warns about it
lingering in your environment.

## Approach 2: JPype Classpath for a Single Class

Use this when you're iterating on one new Java class and don't want to
rebuild MOA at all.

1. Compile your class, linking against the bundled `moa.jar` so that
   `AbstractClassifier` and `Instance` resolve:

   ```bash
   mkdir -p /tmp/myclasses
   javac -cp src/capymoa/jar/moa.jar \
         -d /tmp/myclasses \
         org/capymoa/example/AlwaysPositive.java
   ```

   The source file must live at
   `org/capymoa/example/AlwaysPositive.java`, matching its
   `package org.capymoa.example;` declaration — `javac` requires this.

2. Add the compiled output directory to the classpath **before** importing
   anything from `capymoa`:

   ```python
   import jpype

   jpype.addClassPath("/tmp/myclasses")
   import capymoa  # adds moa.jar and starts the JVM with both on the classpath
   from org.capymoa.example import AlwaysPositive
   ```

   `jpype.addClassPath()` just accumulates entries; it doesn't start the
   JVM. So when `import capymoa` runs `_start_jpype()` right after, it
   sees the JVM isn't started yet, adds `moa.jar` on top of your entry, and
   starts the JVM with both on the classpath.

   **This ordering matters.** Importing *anything* from the `capymoa`
   package first — even `from capymoa.env import capymoa_moa_jar`, just to
   look up the default jar path — runs `capymoa/__init__.py` and starts
   the JVM immediately, before your class is on the classpath. Call
   `jpype.addClassPath()` first, in the same process, before touching
   `capymoa` at all.

## Wrap It in Python and Run It

Once your Java class is importable — via either approach — wrap it the
same way any other MOA-backed learner is wrapped: subclass
`capymoa.base.MOAClassifier` and pass the Java class as `moa_learner`. See
[`src/capymoa/classifier/_sgt.py`](https://github.com/adaptive-machine-learning/CapyMOA/blob/main/src/capymoa/classifier/_sgt.py)
for a real, minimal example of this pattern.

```python
from capymoa.base import MOAClassifier
from capymoa.datasets import ElectricityTiny
from capymoa.evaluation import prequential_evaluation


class AlwaysPositiveClassifier(MOAClassifier):
    def __init__(self, schema):
        super().__init__(moa_learner=AlwaysPositive, schema=schema)


stream = ElectricityTiny()
learner = AlwaysPositiveClassifier(stream.get_schema())
results = prequential_evaluation(stream, learner, max_instances=1000)
print(results["cumulative"].accuracy())
```

From here, follow the [FAQ](faq.md) for where the permanent wrapper file
should live, which base class to use for regressors/anomaly detectors, and
how to add tests and docstrings. Once your Java change is merged upstream
in MOA, follow the [Update MOA guide](update_moa.md) to bump the version
CapyMOA bundles.
