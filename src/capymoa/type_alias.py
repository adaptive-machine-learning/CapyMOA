from numpy import double
from numpy.typing import NDArray

FeatureVector = NDArray[double]
"""
Type definition for a feature vector, which is represented as a one dimensional
NumPy array of double precision floating-point numbers.
"""

LabelIndex = int
"""
Type definition for a class-label index, which is a non-negative integer that is
the index of the label in a list of labels.
"""

LabelProbabilities = NDArray[double]
"""
Type definition for a prediction probability, which is represented as a one
dimensional NumPy array of double precision floating-point numbers.
"""

Label = str
"""
Type definition for a class label.
"""

TargetValue = double
"""
Alias for a dependent variable in a regression task.
"""
