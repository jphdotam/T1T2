from utils.cfg import load_config
from utils.dataset import T1T2Dataset
from utils.transforms import get_segmentation_transforms

CONFIG = "experiments/028.yaml"

# Load config
cfg, model_dir = load_config(CONFIG)

# Data
train_transforms, test_transforms = get_segmentation_transforms(cfg)
ds_train = T1T2Dataset(cfg, 'train', train_transforms)
ds_test = T1T2Dataset(cfg, 'test', test_transforms)

sequences_train = list(set([s.rsplit('__',2)[0] for s in ds_train.sequences]))
sequences_test = list(set([s.rsplit('__',2)[0] for s in ds_test.sequences]))

patients_train = list(set([s.rsplit('_',2)[0] for s in sequences_train]))
patients_test = list(set([s.rsplit('_',2)[0] for s in sequences_test]))

print(print(f"{len(patients_train)} training patients, {len(patients_test)} testing patients"))
