import numpy as np
#this file should be modified too becasue the linear.py is using initializer.py

def xavier_initializer(shape):
    """Xavier initialization scales the weights by sqrt(1 / n_in), 
    where n_in is the number of input units (fan_in)."""
    fan_in = shape[0]
    scale = np.sqrt(1 / fan_in)
    return np.random.uniform(-scale, scale, size=shape)

def he_initializer(shape):
    """He initialization scales the weights by sqrt(2 / n_in), 
    where n_in is the number of input units (fan_in)."""
    fan_in = shape[0]
    scale = np.sqrt(2 / fan_in)
    return np.random.randn(*shape) * scale

def random_normal_initializer(shape, mean=0.0, stddev=0.05):
    """Random normal initialization with specified mean and standard deviation."""
    return np.random.normal(loc=mean, scale=stddev, size=shape)

def zero_initializer(shape):
    """Initialize weights to zero."""
    return np.zeros(shape)

def one_initializer(shape):
    """Initialize weights to one."""
    return np.ones(shape, dtype=np.float64)

def initializer(shape, mode="xavier"):
    """Select an initialization method based on mode."""
    if mode == "xavier":
        return xavier_initializer(shape)
    elif mode == "he":
        return he_initializer(shape)
    elif mode == "random_normal":
        return random_normal_initializer(shape)
    elif mode == "zero":
        return zero_initializer(shape)
    elif mode == "one":
        return one_initializer(shape)
    else:
        raise NotImplementedError("Not implemented initializer method")
