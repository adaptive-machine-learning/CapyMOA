# Implement a method in Java and use it in Python

CapyMOA uses [JPype](https://jpype.readthedocs.io/) to call into the Java
library [MOA](https://moa.cms.waikato.ac.nz/). If you're adding a new
learner backed by a MOA Java class, you'll usually want to write and test
the Java side first, before writing the permanent Python wrapper described
in the [FAQ](faq.md#what-does-a-learner-implement).

This guide walks through getting a Java class onto the classpath so you can
call it from Python, using a small worked example: `AlwaysPositive`, a
trivial `moa.classifiers.Classifier` that always predicts class index 0.
It's compiled against a real MOA interface
(`moa.classifiers.AbstractClassifier`), so the compile and classpath steps
below are the ones you'll actually hit with a real classifier. See MOA's
own [Introduction to the API of
MOA](https://moa.cms.waikato.ac.nz/tutorial-2-introduction-to-the-api-of-moa/)
tutorial for a full walkthrough of `AbstractClassifier` and the methods
implemented below (`trainOnInstanceImpl`, `getVotesForInstance`, and so on).

```java
package org.capymoa.example;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.Measurement;

public class AlwaysPositive extends AbstractClassifier implements Classifier {

    private static final long serialVersionUID = 1L;

    public IntOption targetClassOption = new IntOption(
            "targetClass", 'c',
            "Class index this classifier always predicts.", 0, 0, Integer.MAX_VALUE);

    @Override
    public String getPurposeString() {
        return "A trivial classifier that always predicts one configurable class.";
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
        votes[targetClassOption.getValue()] = 1.0;
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
        out.append("AlwaysPositive: always predicts class ")
                .append(targetClassOption.getValue())
                .append('.');
    }
}
```

`targetClassOption` uses MOA's own options system
(`com.github.javacliparser`), the same mechanism every built-in MOA
learner uses to expose configurable parameters. Its short flag, `'c'`,
becomes the `-c` argument in a MOA CLI string. Section "Wrap it in Python
and run it" below sets this option from Python.

There are two ways to get a class like this onto CapyMOA's classpath:
build all of MOA, or add just this one class to the classpath. Prefer
building MOA if you plan to upstream your change into the MOA project. Use
the classpath approach if you want something lighter weight for prototypes
or standalone research.

## Background: how CapyMOA loads MOA

`import capymoa` starts the JVM as the very first thing it does
(`src/capymoa/__init__.py`), before any other submodule is imported. That
startup, in `src/capymoa/_prepare_jpype.py::_start_jpype()`, does the
following:

1. Resolves the MOA jar via `capymoa_moa_jar()` (`src/capymoa/env.py`): the
   bundled `src/capymoa/jar/moa.jar` by default, or the `CAPYMOA_MOA_JAR`
   environment variable if it's set.
2. Adds that jar to the classpath with `jpype.addClassPath(moa_jar)`.
3. Starts the JVM with `jpype.startJVM(...)`, without passing an explicit
   `classpath` argument. JPype then builds the classpath itself from every
   path added with `jpype.addClassPath()`, plus the standard `CLASSPATH`
   environment variable.

It skips all of this if `jpype.isJVMStarted()` is already `True`.

Two details matter for adding your own class:

* CapyMOA never overrides the classpath explicitly, so the `CLASSPATH`
  environment variable reaches the JVM untouched. Approach 2 uses this.
* CapyMOA skips its own setup once the JVM is running, so
  `jpype.addClassPath()` calls made before `import capymoa` also reach the
  JVM. This is a fallback for when you can't set an environment variable.

## Approach 1: build the whole MOA project

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
so CapyMOA goes back to using the bundled jar.

## Approach 2: put a single class on the classpath

Use this when you're iterating on one new Java class and don't want to
rebuild MOA at all.

1. Compile your class, linking against the bundled `moa.jar` so that
   `AbstractClassifier` resolves:

   ```bash
   mkdir -p /tmp/myclasses
   javac -cp src/capymoa/jar/moa.jar \
         -d /tmp/myclasses \
         org/capymoa/example/AlwaysPositive.java
   ```

   The source file must live at `org/capymoa/example/AlwaysPositive.java`,
   matching its `package org.capymoa.example;` declaration. `javac`
   requires this layout.

2. Add the compiled output directory to the standard `CLASSPATH`
   environment variable, then run Python as usual:

   ```bash
   export CLASSPATH=/tmp/myclasses
   python -c "
   import capymoa
   from org.capymoa.example import AlwaysPositive
   print(AlwaysPositive)
   "
   ```

   CapyMOA never passes an explicit `classpath` argument to
   `jpype.startJVM()`, so JPype falls back to its own default: every path
   added with `jpype.addClassPath()`, plus the `CLASSPATH` environment
   variable. Setting `CLASSPATH` is enough. There's no extra setup code,
   and import order doesn't matter.

   If you'd rather not set an environment variable,
   `jpype.addClassPath()` works too, but only if called before
   `import capymoa`. Importing any part of the `capymoa` package starts
   the JVM immediately, and JPype cannot extend the classpath once the
   JVM is running:

   ```python
   import jpype

   jpype.addClassPath("/tmp/myclasses")
   import capymoa  # must come after addClassPath
   from org.capymoa.example import AlwaysPositive
   ```

## Wrap it in Python and run it

Once your Java class is importable, wrap it the same way any other
MOA-backed learner is wrapped: subclass `capymoa.base.MOAClassifier` and
pass the Java class as `moa_learner`. Pass a MOA CLI string through the
`CLI` argument to set options like `targetClassOption`; CapyMOA forwards it
to `moa_learner.getOptions().setViaCLIString(CLI)`. See
{py:class}`capymoa.classifier.StochasticGradientTree` for a real classifier
that configures several options this way.

```python
from capymoa.base import MOAClassifier
from capymoa.datasets import ElectricityTiny
from capymoa.evaluation import prequential_evaluation


class AlwaysPositiveClassifier(MOAClassifier):
    def __init__(self, schema, target_class: int = 0):
        super().__init__(
            moa_learner=AlwaysPositive,
            schema=schema,
            CLI=f"-c {target_class}",
        )


stream = ElectricityTiny()
learner = AlwaysPositiveClassifier(stream.get_schema(), target_class=1)
results = prequential_evaluation(stream, learner, max_instances=1000)
print(results["cumulative"].accuracy())
```

From here, follow the [FAQ](faq.md) for where the permanent wrapper file
should live, which base class to use for regressors and anomaly detectors,
and how to add tests and docstrings. Once your Java change is merged
upstream in MOA, follow the [Update MOA guide](update_moa.md) to bump the
version CapyMOA bundles.
