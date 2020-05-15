import os
import yaml

def load_config(configpath):
    with open(configpath) as f:
        cfg = yaml.safe_load(f)

    experiment_id = os.path.splitext(os.path.basename(configpath))[0]
    cfg['experiment_id'] = experiment_id

    vis_dir = os.path.join(cfg['output']['vis_dir'], experiment_id)
    model_dir = os.path.join(cfg['output']['model_dir'], experiment_id)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return cfg, vis_dir, model_dir
