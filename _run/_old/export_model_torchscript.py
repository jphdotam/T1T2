import os
import torch
from torch.utils.data import DataLoader

from utils.cfg import load_config
from utils.dataset import T1T2Dataset
from utils.models import load_seg_model
from utils.transforms import get_segmentation_transforms

CONFIG = "../experiments/009.yaml"
MODEL_PATH = "../../output/models/009/1_250_0.0061166.pt"
OUT_DIR = "../../output/models/009/"

# Load config & ensure not data parallel
cfg, vis_dir, model_dir = load_config(CONFIG)
cfg['training']['dataparallel'] = False
cfg['training']['mixed_precision'] = False

# Load model
model, starting_epoch, state = load_seg_model(cfg)
model.eval()
model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])

# Data
train_transforms, _ = get_segmentation_transforms(cfg)
ds_train = T1T2Dataset(cfg, 'train', train_transforms, fold=1)
dl_train = DataLoader(ds_train, cfg['training']['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
x, y = next(iter(dl_train))

# Trace
traced_model = torch.jit.trace(model, x, strict=False)
traced_model.save(os.path.join(OUT_DIR, "t1t2_fcn.zip"))

# def network_to_half(model):
#     """
#     Convert model to half precision in a batchnorm-safe way.
#     """
#     def bn_to_float(module):
#         """
#         BatchNorm layers need parameters in single precision. Find all layers and convert
#         them back to float.
#         """
#         if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
#             module.float()
#         for child in module.children():
#             bn_to_float(child)
#         return module
#     return bn_to_float(model.half())
#
# model = network_to_half(model)
# traced_model = torch.jit.trace(model, x.half(), strict=False)
# traced_model.save(os.path.join(OUT_DIR, "t1t2_fcn_fp16.zip"))