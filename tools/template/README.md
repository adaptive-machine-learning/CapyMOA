# MOA to CapyMOA Classifier Template

The template tool has some additional dependencies:
```bash
pip install jinja2 click
```

To generate a CapyMOA wrapper for the MOA classifier `moa.classifiers.trees.PLASTIC`, run the following command:

```bash
python main.py "moa.classifiers.trees.PLASTIC"
```

Regression models and other object types are not yet supported.
