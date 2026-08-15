
import warnings

from ..cnn.bricks.wrappers import Conv2d, ConvTranspose2d, Linear, MaxPool2d

class Conv2d_deprecated(Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            
            )

class ConvTranspose2d_deprecated(ConvTranspose2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            
            )

class MaxPool2d_deprecated(MaxPool2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            
            )

class Linear_deprecated(Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            
            )
