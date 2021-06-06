import numpy as np

def cross_entropy(category, probability):
  # Why using np.float_? Answer is bellow from StackOverflow
    cat, prob = np.float_(category), np.float_(probability)
    return -np.sum(cat * np.log(prob) + (1 - cat) * np.log(1 - prob))
  
""" by Steven Rumbalski - https://stackoverflow.com/questions/6205020/numpy-types-with-underscore-int-float-etc
Names for the data types that would clash with standard Python object names are followed by a trailing underscore, ’ ’. 
These data types are so named because they use the same underlying precision as the corresponding Python data types.

The array types bool_, int_, complex_, float_, object_, unicode_, and str_ are enhanced-scalars. They are very similar to the standard Python types 
(without the trailing underscore) and inherit from them (except for bool_ and object_). They can be used in place of the standard Python types whenever desired. 
Whenever a data type is required, as an argument, the standard Python types are recognized as well.
"""
