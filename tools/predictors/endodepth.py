import torch
import json
from types import SimpleNamespace
import torchvision.transforms as tfm

import networks.layers as layers
from tools.predictor import Predictor
from tools.trainers.endodepth import EndoDepthTrainer


class EndoDepthPreditor(Predictor):

    def __init__(self, options, model_path: str = None, data_loader=None, *args, **kwargs):

        if isinstance(options, str):
            assert isinstance(options, str), "Arg 'options' should be path of the opt.json."
            with open(options, 'r') as f:
                options = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        else:
            assert(isinstance(options, SimpleNamespace))

        if model_path is not None:
            options.load_weights_folder = model_path

        models, _ = EndoDepthTrainer.make_models(options)

        self.transform = tfm.Compose([
            tfm.ToTensor(),
            tfm.Resize((options.height, options.width)),
            tfm.Normalize((0.5), (0.5))
        ])

        super(EndoDepthPreditor, self).__init__(
            models=models,
            loader=data_loader,
            options=options,
            *args, **kwargs
        )

    def on_model_input(self, inputs):
        return inputs["color", 0]

    def on_model_output(self, outputs):
        return outputs

    def run_batch(self, inputs):
        input = self.on_model_input(inputs)

        features = self.models['encoder'](input)
        output = self.models['depth'](features)
        output = self.on_model_output(output)

        losses = self.compute_loss(inputs, output)
        return output, losses

    def on_load_model(self, state_dict):
        if 'height' in state_dict.keys():
            self.options.height = state_dict['height']
        if 'width' in state_dict.keys():
            self.options.width = state_dict['width']

    def predict(self, input: torch.Tensor):
        """ Pass a minibatch through the network and generate images and losses
        """
        input = input.to(self.device)
        features = self.models["encoder"](input)
        outputs = self.models["depth"](features)

        # for scale in self.options.scales:
        disp = outputs[0]
        # Convert sigmoid output to depth
        _, depth = layers.disp_to_depth_log10(
            disp, self.options.min_depth_units,
            self.options.max_depth_units, 1.0)

        return depth
