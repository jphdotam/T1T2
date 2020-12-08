import os
import torch
from torch.utils.data import DataLoader

from utils.cfg import load_config
from utils.dataset import T1T2Dataset
from utils.models import load_seg_model
from utils.transforms import get_segmentation_transforms

CONFIG = "./experiments/028.yaml"
MODEL_PATH = "./output/models/028/150_0.0009576.pt"

# Load config & ensure not data parallel
cfg, model_dir = load_config(CONFIG)
cfg['training']['dataparallel'] = False
cfg['training']['mixed_precision'] = False

# Load model
model, starting_epoch, state = load_seg_model(cfg)
model.eval()
model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])

# Data
_, test_transforms = get_segmentation_transforms(cfg)
ds_test = T1T2Dataset(cfg, 'test', test_transforms)
dl_train = DataLoader(ds_test, 4, shuffle=True, num_workers=0, pin_memory=True)
x, y, _filepath = next(iter(dl_train))

torch.onnx.export(model,
                  x.cuda(),
                  MODEL_PATH + ".onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})
