# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
from collections import OrderedDict

import os
import time
from typing import Iterable
import torch
import json
import numpy as np
from torch import optim
from tqdm import tqdm
from abc import abstractmethod
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from tools.utils import sec_to_hm_str

# TODO: update verbose output


class Trainer:

    @staticmethod
    @abstractmethod
    def make_models(options, *args, **kwargs):
        """ Maker your models dict with this function, 
            this will be used by predictor.
            Return (model:dict, param:list)
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_loss(self, inputs, outputs):
        """
        Return a dict of losses, at least item called 'loss' is needed.
        Args:
            inputs ([type]): [description]
            outputs ([type]): [description]
        """
        raise NotImplementedError()

    @abstractmethod
    def on_save_model(self, state_dict):
        """ Update state_dict to save.
        """
        pass

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
    def on_model_output(self, outputs):
        """ Update default outputs for model wrapping.
            This will work if run_batch() is not custom-implemented. 
        """
        return outputs

    @abstractmethod
    def run_batch(self, inputs, bid=None):
        """ Process batch with network and get outputs and losses.
            You should return (outputs, losses) as the result.
        """
        return self.__run_batch(inputs)  # Built-in run_batch method.

    @abstractmethod
    def on_val_ending(self, losses):
        return losses

    def __init__(self,
                 models: OrderedDict,
                 optim: Optimizer,
                 train_loader: DataLoader,
                 sceduler: lr_scheduler._LRScheduler = None,
                 val_loader: DataLoader = None,
                 verbose: int = 0,
                 sequential: bool = True,
                 losses: dict = None,
                 options=None):
        """[summary]

        Args:
            models (OrderedDict): [description]
            optim (Optimizer): [description]
            train_loader (DataLoader): [description]
            sceduler (lr_scheduler._LRScheduler, optional): [description]. Defaults to None.
            val_loader (DataLoader, optional): [description]. Defaults to None.
            verbose (int, optional): [description]. Defaults to 0.
            sequential (bool, optional): If models is prallel or sequential. Defaults to True.
            losses (dict, Optinal): Name of losses for each optimizer.
            options ([type], optional): [description]. Defaults to None.
        """

        self.models = models

        # Get optimizer
        self.optims: dict = {}
        if isinstance(optim, dict):
            self.optims = optim
        elif isinstance(optim, (list, tuple)):
            for i, v in enumerate(optim):
                self.optims[i] = v
        elif isinstance(optim, Optimizer):
            self.optims[0] = optim
        else:
            raise NotImplementedError()

        self.scedulers = sceduler if isinstance(sceduler, (list, tuple)) else [sceduler]
        # Remove None object
        self.optims = {k: v for k, v in self.optims.items() if v is not None}
        self.scedulers = tuple(filter(None, self.scedulers))

        self.losses = losses
        if losses is not None:
            # Check names
            for k in self.optims.keys():
                assert k in losses.keys()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.options = options
        self.verbose = verbose
        self.sequential = sequential

        assert(self.options.model_name is not None)

        self.device = torch.device("cpu" if self.options.no_cuda else "cuda")
        self.log_path = os.path.join(self.options.log_dir, self.options.model_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # Move mode to device
        self.set_device()

        ######################################################
        # TRAIN CONTROL #
        ######################################################
        self.epoch = 0
        self.step = 0

        if self.options.load_weights_folder is None:
            self.options.load_weights_folder = os.path.join(self.log_path, "models")

        self.no_batch = len(self.train_loader)
        self.no_steps = self.no_batch // self.options.batch_size * self.options.num_epochs

        if self.verbose > 0:
            print("Training model named:\n    ", self.options.model_name)
            print("Model modules:")
            for k in self.models:
                print("    {}".format(k))
            print("Optimizer:")
            for k in self.optims:
                print("    {}".format(k))
            print("Models and tensorboard events files are saved to:\n    ", self.options.log_dir)
            print("Training is using:\n    ", self.device)
            print("Train batch size:\n    {}".format(len(train_loader)))
            if val_loader is not None:
                print("Val batch size:\n    {}".format(len(val_loader)))
            else:
                print("Not use validation.")

        if verbose > 0:
            # Display model param size
            print("Network param:")
            for name, model in self.models.items():
                num_params = 0
                for param in model.parameters():
                    num_params += param.numel()
                if verbose > 2:
                    print(model)
                print('    %s: %.3f M' % (name, num_params / 1e6))

        self.load_model(self.options.load_weights_folder)

        ######################################################
        # SAVER & LOGGER #
        ######################################################
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.save_opts()
        file = self.save_model(0)  # test save model
        os.remove(file)

    def set_device(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.to(self.device)

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        if self.epoch is None:
            self.epoch = 0
        self.t0 = time.time()

        with torch.no_grad():
            for opm in self.optims.values():
                opm.step()

        while self.epoch < self.options.num_epochs:
            self.run_epoch()
            self.epoch += 1
            if self.epoch % self.options.save_frequency == 0:
                self.save_model(self.epoch)
            self.val()

    def val(self):
        self.set_eval()

        if self.val_loader is not None:
            i = 0
            with torch.no_grad():
                __losses = {}
                iter = tqdm(self.val_loader) if self.verbose > 0 else self.val_loader
                for __inputs in iter:
                    # Move to target device
                    for key, ipt in __inputs.items():
                        if isinstance(ipt, torch.Tensor):
                            __inputs[key] = ipt.to(self.device)
                        # TODO: add val fucntions
                    _, __loss = self.run_batch(__inputs, None)

                    # Build dict
                    for k, v in __loss.items():
                        if i == 0:
                            __losses[k] = v
                        else:
                            __losses[k] += v
                    i += 1
                # average batches
                for k, v in __losses.items():
                    __losses[k] = v / (i + 1)

                # Maybe to run extra validation
                __losses = self.on_val_ending(__losses)

                # End bactches and print result
                if self.verbose > 1:
                    print('Validate result:')
                for k, v in __losses.items():
                    if self.verbose > 1:
                        print("    {}: {}".format(k, __losses[k]))

                self.log("val", __losses)
                del __inputs, __losses, __loss

        self.set_train()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        self.set_train()

        for sch in self.scedulers:
            sch.step(self.epoch)

        if self.verbose > 0:
            print("Epoch starting lr:")
            for k, opm in self.optims.items():
                print("    {}: {}".format(k, opm.param_groups[0]['lr']))

        for bid, inputs in enumerate(self.train_loader):

            self.step += 1
            t1 = time.time()
            # Move to target device
            for key, ipt in inputs.items():
                if isinstance(ipt, torch.Tensor):
                    inputs[key] = ipt.to(self.device)

            # Run the model
            _, losses = self.run_batch(inputs, bid + 1)

            if self.losses is not None:
                for k, opm in self.optims.items():
                    opm.zero_grad()
                    __kloss = self.losses[k]
                    if isinstance(__kloss, str):
                        losses[__kloss].backward()
                    elif isinstance(__kloss, (list, tuple)):
                        for v in __kloss:
                            losses[v].backward()
                    else:
                        raise NotImplementedError()
                    opm.step()
                    losses['lr/{}'.format(k)] = opm.param_groups[0]['lr']
            else:
                for opm in self.optims.values():
                    opm.zero_grad()
                losses['loss'].backward()
                for k, opm in self.optims.items():
                    opm.step()
                    losses['lr/{}'.format(k)] = opm.param_groups[0]['lr']

            # Make log
            dt = time.time() - t1
            if bid % self.options.log_frequency == 0:
                self.log_time(bid, dt, losses["loss"].item())
                self.log("train", losses)

    def __run_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        input = self.on_model_input(inputs)
        if self.sequential:
            # Sequential mode
            output = input
            for __name, __model in self.models.items():
                output = __model(output)
            output = self.on_model_output(output)
            losses = self.compute_loss(inputs, output)
            return output, losses
        else:
            # Parallel mode
            outputs = {}
            for __name, __model in self.models.items():
                outputs[__name] = __model(input)
            outputs = self.on_model_output(outputs)
            losses = self.compute_loss(inputs, outputs)
            return outputs, losses

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.options.batch_size / duration
        time_sofar = time.time() - self.t0
        print_string = "epoch {:>3} | batch {:>3}/{:>3} | examples/s: {:5.1f} | loss: {:.5f} | time elapsed: {}"
        print(print_string.format(self.epoch, batch_idx, self.no_batch,
                                  samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar)))

    def log(self, mode, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.options.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

        if self.verbose > 1:
            opt = json.dumps(self.options.__dict__, indent=2)
            print("Options:")
            print(opt)

    def save_model(self, epoch, extra=None):
        """Save model weights to disk
        """
        state_dict = {
            'epoch': epoch,
            'step': self.step,
            'model': {},
            'optim': {}
        }
        for __name, __model in self.models.items():
            state_dict['model'][__name] = __model.state_dict()

        for k, opm in self.optims.items():
            state_dict['optim'][k] = opm.state_dict()

        res = self.on_save_model(state_dict)
        state_dict = res if res is not None else state_dict

        if extra is None:
            save_path = os.path.join(self.log_path, "models", "{}_{:04d}.pt".format(self.options.model_name, epoch))
        else:
            save_path = os.path.join(self.log_path, "models", "{}_{:04d}_{}.pt".format(self.options.model_name, epoch, extra))
        torch.save(state_dict, save_path)
        return save_path

    @staticmethod
    def get_model_path(file=None):
        if file is None:
            return None
        file = os.path.expanduser(file)
        if not os.path.exists(file):
            return None

        # Check is file or directory
        if os.path.isfile(file):
            return file
        elif os.path.isdir(file):
            # Retrieve all files and load newest checkpoint
            files = []
            files = [fn for fn in os.listdir(file) if fn.endswith(".pt")]
            files.sort()
            if len(files) > 0:
                return os.path.join(file, files[-1])
        return None

    def load_model(self, file=None):
        """Load model(s) from disk
        """
        file = self.get_model_path(file)
        if file is None:
            return

        # Check model file
        assert os.path.isfile(file)
        print("Loading model:\n    {}".format(file))

        # Load weights
        state_dict = torch.load(file)

        for __k, _ in self.optims.items():
            if __k in state_dict['optim'].keys():
                self.optims[__k].load_state_dict(state_dict['optim'][__k])
            else:
                print("    Not found optimizer for {}".format(__k))

        for __n, _ in self.models.items():
            if __n in state_dict['model'].keys():
                self.models[__n].load_state_dict(state_dict['model'][__n])
            else:
                print("    Not found weights for {}".format(__n))

        if 'epoch' in state_dict.keys():
            self.epoch = state_dict['epoch']
        if 'step' in state_dict.keys():
            self.step = state_dict['step']

        self.on_load_model(state_dict)
