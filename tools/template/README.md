# MOA to CapyMOA Classifier Template

The template tool has some additional dependencies:
```bash
pip install jinja2 click
```

To generate a CapyMOA wrapper for the MOA classifier `moa.classifiers.trees.PLASTIC`, run the following command:

```bash
python main.py "moa.classifiers.trees.PLASTIC"
```

The wrapper passes options as a dict, so class options accept CLI strings, CapyMOA wrappers, or Java objects.

Regression models and other object types are not yet supported.
