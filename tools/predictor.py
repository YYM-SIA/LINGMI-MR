from genericpath import isdir
import os
import time
import torch
import json
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from abc import abstractmethod
from torch import jit
from torch.utils.data import DataLoader

# TODO: update verbose output


class Predictor:

    @abstractmethod
    def on_load_model(self, state_dict):
        """ Custom-defined items from state_dict.
        """
        pass

    @abstractmethod
    def on_model_input(self, inputs):
        """ Replace default inputs for model wrapping.
            This will work if run_batch() is not custom-implemented. 
        """
        return inputs

    @abstractmethod
    def on_predict(self, inputs, predicts):
        """ Do something with predictions.
        """
        pass

    @abstractmethod
    def run_batch(self, inputs):
        """ Process batch with network and get outputs.
            You should return outputs as the result.
        """
        return self.__run_batch(inputs)  # Built-in run_batch method.

    def __init__(self,
                 models: OrderedDict,
                 loader: DataLoader = None,
                 verbose: int = 0,
                 sequential: bool = True,
                 options=None):

        self.models = models
        self.loader = loader
        self.options = options
        self.verbose = verbose
        self.sequential = sequential

        self.device = torch.device("cpu" if self.options.no_cuda else "cuda")
        # Check model loading path
        self.log_path = os.path.join(self.options.log_dir, self.options.model_name)
        if not os.path.exists(self.log_path):
            self.log_path = None

        # Move mode to device
        self.set_device()
        self.set_eval()

        ######################################################
        # LOAD WEIGHT #
        ######################################################
        if self.options.load_weights_folder is None and self.log_path is not None:
            self.options.load_weights_folder = os.path.join(self.log_path, "models")
        self.load_model(self.options.load_weights_folder)

        if self.verbose > 0:
            print("Model named:\n    ", self.options.model_name)
            print("Using device:\n    ", self.device)

    def set_device(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.to(self.device)

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def __run_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        input = self.on_model_input(inputs)
        if self.sequential:
            # Sequential mode
            output = input
            for __name, __model in self.models.items():
                output = __model(output)
            return output
        else:
            # Parallel mode
            outputs = {}
            for __name, __model in self.models.items():
                outputs[__name] = __model(input)
            return outputs

    def predict(self):
        if self.loader is None:
            raise NotImplementedError()

        with torch.no_grad():
            size = len(self.loader)
            iter = tqdm(self.loader) if self.verbose > 0 else self.loader

            if self.verbose > 0:
                t0 = time.time()

            for inputs in iter:
                # Move to target device
                for key, ipt in inputs.items():
                    if isinstance(ipt, torch.Tensor):
                        inputs[key] = ipt.to(self.device)

                output = self.run_batch(inputs)
                self.on_predict(inputs, output)

            if self.verbose > 0:
                dt = time.time() - t0
                print("Total time: {}\nAverage time: {}\nSpeed {} samp/s".format(
                    dt, dt / size, size / dt))

    def load_model(self, file=None):
        """Load model(s) from disk
        """
        if file is None:
            return
        file = os.path.expanduser(file)
        if not os.path.exists(file):
            raise FileNotFoundError()

        # Check is file or directory
        if os.path.isdir(file):
            # Retrieve all files and load newest checkpoint
            files = []
            files = [fn for fn in os.listdir(file) if fn.endswith(".pt")]
            files.sort()
            if len(files) > 0:
                self.load_model(os.path.join(file, files[-1]))
            return

        # Check model file
        assert os.path.isfile(file)
        if self.verbose > 0:
            print("Loading model:\n    {}".format(file))

        # Load weights
        state_dict = torch.load(file)

        for __n, __model in self.models.items():
            if __n in state_dict['model'].keys():
                __model.load_state_dict(state_dict['model'][__n])
            else:
                print("    Not found weights for {}".format(__n))

        self.on_load_model(state_dict)

    def dump(self, path):
        assert(os.path.isdir(path))
        for __name, __model in self.models.items():
            __jit = jit.script(__model)
            __jit.save(os.path.join(path, "{}.pt".format(__name)))
