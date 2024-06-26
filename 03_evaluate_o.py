"""
Evaluate on out-of-order functions
"""
import os
import torch
import glob

from synthetic.init import set_seed, read_config
from synthetic.generator import SyntheticEvalCombinatorial
from net.nanogpt import nanoGPT


def load_net(fname):
    ckpt = torch.load(fname)
    net_cfg = ckpt['config']
    
    net = nanoGPT(net_cfg.net)
    net.load_state_dict(ckpt['net'])
    return net, net_cfg


def fetch_last_ckpt(cfg):
    ckpt_dir = os.path.join("./ckpts", cfg.ckpt_tag, "*")

    def itr(ck):
        return int((ck.split("_")[-1]).split(".")[0])

    all_dirs = [(itr(ck), ck) for ck in glob.glob(ckpt_dir)]
    all_dirs = sorted(all_dirs)
    return all_dirs[-1][1]


def main(cfg):
    set_seed(cfg.seed)

    ckpt_file = fetch_last_ckpt(cfg)
    _, net_cfg = load_net(ckpt_file)

    evaluator = SyntheticEvalCombinatorial(net_cfg, cfg.nsamples, cfg.nbatch)
    net, _ = load_net(ckpt_file)
    mat = evaluator.get_acc(net)
    evaluator.save_accs(cfg, mat)


if __name__ == "__main__":
    cfg = read_config("./config/eval/conf_o.yaml")
    main(cfg)
