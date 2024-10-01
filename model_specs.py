from random import choice
from copy import deepcopy
from typing import List
from itertools import product
import yaml


# TODO: Clarify with Raghav whether input features is a model characteristic or a dataset characteristic

"""Model topology specifications"""
MODEL_SPEC_VERSION = 1 # Optionally increment this if the model spec format changes
INITIAL_CHANNELS = [1, 2, 3, 4] # How many input channels
MODEL_DEPTHS = [1, 2, 3, 4] # How many layers of the UNet
#QUANTIZATION_LEVELS = [2,4,8,16,32] # Assumes integer weights.
KERNEL_SIZES = [1, 3, 5] # Kernel size for convolutional layers
#TODO: Potentially add activation function?
#TODO: Should we support different feature scaling per layer?

class UNetSpec(yaml.YAMLObject):
    yaml_tag = u'!UNetSpec'
    depth: int
    #quantization_level: int
    kernel_sizes: List[int]
    initial_channels: int
    def __init__(self, dict=None):
        if dict is not None:
            self.depth = dict['depth']
            #self.quantization_level = dict['quantization_level']
            self.kernel_sizes = dict['kernel_sizes']
            self.initial_channels = dict['initial_channels']
    def __repr__(self):
        return f"UNetSpec(depth={self.depth}, kernel_sizes={self.kernel_sizes}, initial_channels={self.initial_channels})"

def sample_one():
    spec = UNetSpec()
    spec.depth = choice(MODEL_DEPTHS)
    #spec.quantization_level = choice(QUANTIZATION_LEVELS)
    spec.kernel_sizes = [choice(KERNEL_SIZES) for _ in range(spec.depth + 1)] # One kernel size per layer + one for the middle
    spec.initial_channels = choice(INITIAL_CHANNELS)
    return spec

def sample(n=1):
    """Generate n random model specifications"""
    specs = []
    for _ in range(n):
        spec = sample_one()
        specs.append(deepcopy(spec))
    return specs

def ancestral_sampling_without_replacement(n=1):
    """Generate n random model specifications without replacement"""
    spec_set = set()
    for _ in range(n):
        spec = sample_one()

        i = 0 # counter to avoid infinite loop
        while spec in spec_set:
            spec = sample_one()

            i += 1
            if i > 1000:
                raise Exception('Infinite loop detected')

        spec_set.add(spec)
    return list(spec_set)

def combinations_length():
    """Return the number of possible model specifications"""
    return sum([len(KERNEL_SIZES)**(depth + 1) for depth in MODEL_DEPTHS]) # * len(QUANTIZATION_LEVELS)

def model_iterator():
    """Iterate over all possible model specifications"""
    for depth in MODEL_DEPTHS:
        #for quantization_level in QUANTIZATION_LEVELS:
            for kernel_sizes in product(KERNEL_SIZES, repeat=depth+1):
                spec = UNetSpec()
                spec.depth = depth
                #spec.quantization_level = quantization_level
                spec.kernel_sizes = kernel_sizes
                yield spec