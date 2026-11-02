"""
Run the steganogan command line tool with PyTorch >= 2.6.

The pretrained steganogan models are full pickled objects in the legacy
torch format, which torch.load(weights_only=True) (the default since
PyTorch 2.6) can not load. Instead of loading them without restrictions,
they are loaded with an unpickler that only resolves the classes stored in
those models (listed by scanning the pickle opcodes, without loading them).
Any other global in the file raises an error.
"""

import sys
import pickle
import collections

import torch
import torch.nn

import steganogan.models
import steganogan.encoders
import steganogan.decoders
import steganogan.critics


def _get_thnn_function_backend():
    # Removed from PyTorch; the legacy models still reference it
    return None


class _OptimizerPlaceholder:
    """ The pickled Adam optimizers (only used for training) are not
        compatible with current PyTorch: keep their state as plain data """
    def __setstate__(self, state):
        self.__dict__.update(state)


ALLOWED_GLOBALS = {
    "__builtin__.dict": dict,
    "builtins.dict": dict,
    "collections.OrderedDict": collections.OrderedDict,
    "collections.defaultdict": collections.defaultdict,
    "steganogan.models.SteganoGAN": steganogan.models.SteganoGAN,
    "steganogan.encoders.BasicEncoder": steganogan.encoders.BasicEncoder,
    "steganogan.encoders.DenseEncoder": steganogan.encoders.DenseEncoder,
    "steganogan.encoders.ResidualEncoder": steganogan.encoders.ResidualEncoder,
    "steganogan.decoders.DenseDecoder": steganogan.decoders.DenseDecoder,
    "steganogan.critics.BasicCritic": steganogan.critics.BasicCritic,
    "torch.FloatStorage": torch.FloatStorage,
    "torch.LongStorage": torch.LongStorage,
    "torch._utils._rebuild_parameter": torch._utils._rebuild_parameter,
    "torch._utils._rebuild_tensor_v2": torch._utils._rebuild_tensor_v2,
    "torch.device": torch.device,
    "torch.nn.backends.thnn._get_thnn_function_backend": _get_thnn_function_backend,
    "torch.nn.modules.activation.LeakyReLU": torch.nn.LeakyReLU,
    "torch.nn.modules.activation.Tanh": torch.nn.Tanh,
    "torch.nn.modules.batchnorm.BatchNorm2d": torch.nn.BatchNorm2d,
    "torch.nn.modules.container.Sequential": torch.nn.Sequential,
    "torch.nn.modules.conv.Conv2d": torch.nn.Conv2d,
    "torch.optim.adam.Adam": _OptimizerPlaceholder,
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        key = module + "." + name
        if key in ALLOWED_GLOBALS:
            return ALLOWED_GLOBALS[key]
        raise pickle.UnpicklingError("global not allowed: " + key)


class _RestrictedPickleModule:
    """ pickle module for torch.load(): only the allowed globals """
    __name__ = "restricted_pickle"
    Unpickler = RestrictedUnpickler

    @staticmethod
    def load(f, **kwargs):
        return RestrictedUnpickler(f, **kwargs).load()


_torch_load = torch.load

def _restricted_load(f, map_location=None, **kwargs):
    kwargs.pop("weights_only", None)
    kwargs.pop("pickle_module", None)
    return _torch_load(f, map_location=map_location, weights_only=False,
                       pickle_module=_RestrictedPickleModule, **kwargs)

steganogan.models.torch.load = _restricted_load


if __name__ == "__main__":
    from steganogan.cli import main
    sys.argv[0] = "steganogan"
    sys.exit(main())
