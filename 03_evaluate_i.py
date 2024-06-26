"""
Evaluate on out-of-order functions
"""
import numpy as np
import os
import glob
import torch

from synthetic.init import set_seed, read_config
from synthetic.generator import SyntheticEval

from net.nanogpt import nanoGPT
from net.lstm import AutoLstm


def load_net(fname, lstm):
    ckpt = torch.load(fname)
    net_cfg = ckpt['config']
    
    if not lstm:
        net = nanoGPT(net_cfg.net)
    else:
        net = AutoLstm(net_cfg.net)

    net.load_state_dict(ckpt['net'])
    return net, net_cfg


def fetch_dirs(cfg):
    ckpt_dir = os.path.join("./ckpts", cfg.ckpt_tag, "*")

    def itr(ck):
        return int((ck.split("_")[-1]).split(".")[0])

    all_dirs = [(itr(ck), ck) for ck in glob.glob(ckpt_dir)]
    all_dirs = sorted(all_dirs)

    reduced_alldirs = []
    for it, cdir in all_dirs:

        if it >= cfg.xlim[0] and it <= cfg.xlim[1]:
            reduced_alldirs.append((it, cdir))

    elems = np.round(
        np.linspace(0, len(reduced_alldirs) - 1,cfg.nckpts)).astype(int)
    reduced_alldirs = [reduced_alldirs[e] for e in elems]

    return reduced_alldirs


def main(cfg):
    set_seed(cfg.seed)

    sorted_dirs = fetch_dirs(cfg)

    _, net_cfg = load_net(sorted_dirs[0][1], cfg.lstm)

    evaluator = SyntheticEval(net_cfg, cfg.nsamples,
                              cfg.nbatch, cfg.direct_eval,
                              cfg.permute)

    accs = []
    for ck in sorted_dirs:
        net, _ = load_net(ck[1], cfg.lstm)
        mat = evaluator.get_acc(net, cfg.lstm)
        accs.append((ck, mat))

        acc_vals = np.array(list(mat.values()))[:, 0]
        print("Iter: ", ck[0], " Acc: ", np.mean(acc_vals))

    evaluator.save_accs(cfg, accs)


if __name__ == "__main__":
    cfg = read_config("./config/eval/conf.yaml")
    main(cfg)
