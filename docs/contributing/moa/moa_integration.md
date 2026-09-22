# Implement a method in Java and use it in Python

This guide walks through getting a Java class onto the classpath so you can call it from Python, using a small worked example: `AlwaysPositive`, a trivial`moa.classifiers.Classifier` that always predicts one class.
For a full
walkthrough of `AbstractClassifier` and the methods implemented below
(`trainOnInstanceImpl`, `getVotesForInstance`, and so on), see MOA's
[Introduction to the API of
MOA](https://moa.cms.waikato.ac.nz/tutorial-2-introduction-to-the-api-of-moa/)
tutorial.

```java
package example;

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

`targetClassOption` uses MOA's options system
(`com.github.javacliparser`), the same mechanism every built-in MOA
learner uses for configurable parameters. Its short flag, `'c'`, becomes
the `-c` argument in a MOA CLI string.

There are two ways to get a class like this onto CapyMOA's classpath:
build all of MOA, or add just this one class to the classpath. Prefer
building MOA if you plan to upstream your change into the MOA project. Use
the classpath approach for prototypes or standalone research.

## Approach 1: build the whole MOA project

Use this when your Java change spans multiple classes, touches MOA's own
build, or you want the most faithful test.

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

   `capymoa.about()` prints the resolved `CAPYMOA_MOA_JAR` path and a hash
   of the jar, so you can confirm CapyMOA picked up your build:

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
rebuild MOA.

1. Compile your class, linking against the bundled `moa.jar` so that
   `AbstractClassifier` resolves:

   ```bash
   mkdir -p /tmp/myclasses
   javac -cp src/capymoa/jar/moa.jar \
         -d /tmp/myclasses \
         example/AlwaysPositive.java
   ```

   If you use an IDE, you can add a java archive as a dependency through the correct dialogue:
   [InteliJ](https://stackoverflow.com/a/1051705),
   [Eclipse](https://stackoverflow.com/a/5144449),
   [VS Code](https://stackoverflow.com/q/50232557).

   The source file must live at `example/AlwaysPositive.java`, matching its
   `package example;` declaration.

2. Add the compiled output directory to the standard `CLASSPATH`
   environment variable, then run Python as usual:

   ```bash
   export CLASSPATH=/tmp/myclasses
   python -c "
   import capymoa
   from example import AlwaysPositive
   print(AlwaysPositive)
   "
   ```

````{note}
If you prefer not to set an environment variable, call
`jpype.addClassPath()` before `import capymoa` instead. CapyMOA starts the
JVM on its first import, so the call has to come first:

```python
import jpype

jpype.addClassPath("/tmp/myclasses")
import capymoa  # must come after addClassPath
from example import AlwaysPositive
```
````

## Wrap it in Python and run it

Once your Java class is importable, subclass `capymoa.base.MOAClassifier`
and pass the Java class as `moa_learner`. Set options like
`targetClassOption` with a MOA CLI string passed as `CLI`. See
{py:class}`capymoa.classifier.StochasticGradientTree` for a classifier that
configures several options this way.

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
