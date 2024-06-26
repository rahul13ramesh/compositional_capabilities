import torch
import torch.nn.functional as F

from synthetic.init import set_seed, read_config
from synthetic.generator import get_trainLoader, get_evalLoaders
from synthetic.generator import get_space_pos 

from net.nanogpt import nanoGPT
from net.runner import sanity_checks, configure_optimizers
from net.runner import update_cosine_warmup_lr, save_model
from net.runner import log_train, log_eval, move_to_device
from net.runner import evaluate


def main(cfg):
    set_seed(cfg.seed)

    # Get data
    trainLoader = get_trainLoader(cfg)
    evalLoaders = get_evalLoaders(cfg)

    # Check if network is compatible with data
    sanity_checks(cfg, trainLoader)

    # Load network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = nanoGPT(cfg.net)
    net.to(device)
    if cfg.net.compile:
        net = torch.compile(net)
    print("number of parameters: %.2fM" % (net.get_num_params()/ 1e6,))

    # Optimizer
    optimizer = configure_optimizers(net, cfg.optimizer)

    train(cfg, net, (trainLoader, evalLoaders), optimizer, device)


def train(cfg, net, loaders, optimizer, device):

    net.train()
    trainLoader, evalLoaders = loaders

    dt = torch.bfloat16 if cfg.bf16 else torch.float32
    device_info = (device, dt)

    # space_pos is the position of the seperator token. After this token, the
    # transformer should predict the output of the function. We use this to
    # compute the loss only after the seperator code (only during evaluation)
    space_pos = get_space_pos(cfg.data.path, trainLoader)
    
    total_steps = len(trainLoader) * cfg.epochs
    train_loss = []
    lr, it = 0.0, 0

    save_model(cfg, net, optimizer, it)

    print("Total training steps: ", total_steps)
    print("Learning rate warmup steps: ", cfg.optimizer.warmup_iters)

    for _ in range(cfg.epochs):

        for dat, targets in trainLoader:
            if it % cfg.log.eval_interval == 0:
                eval_info = evaluate(net, evalLoaders,
                                     space_pos, device_info)

                log_eval(it, lr, eval_info)
                save_model(cfg, net, optimizer, it)
        
            elif it % cfg.log.log_interval == 0:
                train_loss = log_train(it, lr, train_loss)

            # Update LR
            it, lr = update_cosine_warmup_lr(it, cfg.optimizer,
                                             optimizer, total_steps)

            optimizer.zero_grad(set_to_none=True)
            dat, targets = move_to_device(dat, targets, device)

            # Compute loss
            with torch.amp.autocast(device_type=device, dtype=dt):
                logits = net(dat)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1))

                train_loss.append(loss.item())

            # Update model
            loss.backward()
            if cfg.optimizer.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               cfg.optimizer.grad_clip)

            optimizer.step()

    # Log one final time
    eval_info = evaluate(net, evalLoaders,
                         space_pos, device_info)
    log_eval(it, lr, eval_info)
    save_model(cfg, net, optimizer, it)


if __name__ == "__main__":
    cfg = read_config("./config/train/conf.yaml")
    main(cfg)
