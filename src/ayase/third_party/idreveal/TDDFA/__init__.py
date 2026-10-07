from .TDDFA import TDDFA


def get_mb1(dir_weights = None):
    import os
    import yaml
    dirfile = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(dirfile, 'configs/mb1_120x120.yml')
    cfg = yaml.load(open(config_file), Loader=yaml.SafeLoader)
    if dir_weights is None:
        cfg['checkpoint_fp'] = os.path.join(dirfile, cfg['checkpoint_fp'])
    else:
        cfg['checkpoint_fp'] = os.path.join(dir_weights, os.path.basename(cfg['checkpoint_fp']))
    return cfg

