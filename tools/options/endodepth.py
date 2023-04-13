'''
Author: Wangpeng-512 1119785034@qq.com
Date: 2022-11-04 09:46:20
LastEditors: Wangpeng-512 1119785034@qq.com
LastEditTime: 2022-11-04 17:31:58
FilePath: \code2\tools\options\endodepth.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from tools.option import BasicOptions


class EndoDepthOptions(BasicOptions):
    def __init__(self):
        super().__init__()

        # TRAINING options
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true",
                                 default=False)
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true",
                                 default=False)
        self.parser.add_argument('--encoder', 
                                type=str,   
                                help='type of encoder, base07, large07, tiny07', 
                                default='base07')


class EndoDepthOptions_V3(EndoDepthOptions):
    def __init__(self):
        super().__init__()
        # v3 options
        self.parser.add_argument("--depth_binning",
                                 help="defines how the depth bins are constructed for the cost"
                                 "volume. 'linear' is uniformly sampled in depth space,"
                                 "'inverse' is uniformly sampled in inverse depth space",
                                 type=str,
                                 choices=['linear', 'inverse', 'log'],
                                 default='log'),
        self.parser.add_argument("--num_depth_bins",
                                 type=int,
                                 default=96)
        self.parser.add_argument("--matching_scaler",
                                 help="depth matching scale of input size",
                                 type=int,
                                 default=4)
        self.parser.add_argument("--no_adaptive_bins",
                                 help="depth matching scale of input size"
                                 "Setting adaptive_bins=True will recompute the depth bins used for matching upon each"
                                 "forward pass - this is required for training from monocular video as there is an unknown scale.",
                                 action="store_true",
                                 default=False)
        self.parser.add_argument("--matching_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load for matching",
                                 default=[-1])
        self.parser.add_argument("--no_set_missing_to_max",
                                 action="store_true",
                                 default=False)
        self.parser.add_argument("--no_matching_augmentation",
                                 action='store_true',
                                 help="If set, will not apply static camera augmentation or "
                                      "zero cost volume augmentation during training",
                                 default=False)
        # ###
        self.parser.add_argument("--load_mono_weight",
                                 type=str,
                                 help="monocular depth weight path",
                                 default=None)
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--disable_motion_masking",
                                 help="If set, will not apply consistency loss in regions where"
                                      "the cost volume is deemed untrustworthy",
                                 action="store_true")