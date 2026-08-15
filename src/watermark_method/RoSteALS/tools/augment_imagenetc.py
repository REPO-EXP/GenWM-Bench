
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import random
import numpy as np 
from PIL import Image 

class IdentityAugment(object):
    def __call__(self, x):
        return x 

    def __repr__(self):
        s = f'()'
        return self.__class__.__name__ + s
