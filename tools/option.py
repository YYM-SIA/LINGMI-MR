import os
import argparse


class BasicOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=None)
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))
        self.parser.add_argument("--test_mode",
                                 help="Disable train mode.",
                                 action="store_true")
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="ilsr")

        # DATASET options
        self.parser.add_argument("--png",
                                 help="input image type, default of jpg",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=480)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=480)
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth (normalized with focal length)",
                                 default=0.001)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth (normalized with focal length)",
                                 default=1000.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0])

        # OPTIMIZATION & TRAINING options
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        self.parser.add_argument("--lr_decade_coeff",
                                 type=float,
                                 help="decade coeff of the scheduler",
                                 default=0.1)
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load",
                                 default=None)
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=20)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=2)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
