from tools.trainers.endodepth import plEndoDepth
from tools.options.endodepth import EndoDepthOptions
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import json

options = EndoDepthOptions()
options.parser.add_argument('--config', type=str, help='path to config json', required=True)

if __name__ == "__main__":

    opt = options.parse()
    args_dict = vars(opt)
    with open(opt.config, 'r') as config_file:
        args_dict.update(json.load(config_file))

    model = plEndoDepth(options=opt, verbose=2)

    train_loader = DataLoader(
        model.train_set, opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=False
    )
    # val_loader = DataLoader(
    #     model.val_set, opt.batch_size, shuffle=False,
    #     num_workers=opt.num_workers, pin_memory=True, drop_last=False
    # )

    checkpoint = ModelCheckpoint(monitor="train_loss")
    early_stop = EarlyStopping(monitor="train_loss",
                               min_delta=1e-8, patience=5, mode="min",
                               stopping_threshold=1e-4, divergence_threshold=10, verbose=False)
    trainer = pl.Trainer(gpus=1, max_epochs=opt.num_epochs,
                         precision=32,                         limit_train_batches=0.2,                         limit_train_batches=0.2,
                         callbacks=[checkpoint, early_stop])
    trainer.fit(model, train_loader)

