import os
import random
import itertools
import pickle
import json
import numpy as np
import tqdm
import functools
import torch

from itertools import combinations
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from synthetic.functions import BaseFunction


class SyntheticData:
    """Generates a synthetic sequence of the form
        t, x, t(x)
    """
    def __init__(self, cfg, composed_functions, functions_info):
        self.cfg = cfg
        self.special_tokens = [' ', '<PAD>', 'S']
        self.n_special = len(self.special_tokens)
        self.n_alphabets = cfg.n_alphabets

        self.functions = composed_functions
        self.functions_info = functions_info
        self.task_map()

        self.fdir = 'data/' + cfg.tag

    def task_map(self):
        self.task_idx = {}
        self.task = {}

        if self.cfg.task_tokens:
            self.depth = len(self.functions_info['functions'])
            self.nfuncs = [len(fn) for fn in self.functions_info['functions']]
            self.n_tasks = sum(self.nfuncs)
        else:
            self.n_tasks = 0

        self.nsplit_tasks = {
            'train': len(self.functions_info['train_id']),
            'all': sum(self.nfuncs)
        }

        for dep, nf in enumerate(self.nfuncs):
            for tid in range(nf):
                idx = sum(self.nfuncs[:dep]) + tid + self.n_alphabets
                self.task[idx] = (dep, tid)
                self.task_idx[(dep, tid)] = idx

    def init_tokens(self):
        """
        Initialize the set of tokens and store it into dictionary.
        """
        self.token = {}
        self.token_idx = {}

        # Alphabet tokens
        for i in range(self.n_alphabets):
            self.token[i] = 'X' + str(i)
            self.token_idx['X' + str(i)] = i

        # Task tokens
        for i in range(self.n_tasks):
            idx = i + self.n_alphabets
            task_str = 'T' + str(self.task[idx][0]) + '_' + str(self.task[idx][1])
            self.token[idx] = task_str
            self.token_idx[task_str] = idx

        # Special tokens
        for i in range(len(self.special_tokens)):
            idx = i + self.n_alphabets + self.n_tasks
            self.token[idx] = self.special_tokens[i]
            self.token_idx[self.special_tokens[i]] = idx

    def sample_task(self, split='train'):
        idx = np.random.randint(0, self.nsplit_tasks[split])
        return self.functions[split][idx]

    def sample_token(self):
        # Sample tokens without replacement from [0, self.n_alphabets]
        alph = np.arange(self.n_alphabets)
        tokens = np.random.choice(alph, size=self.cfg.seq_len,
                                  replace=self.cfg.with_replacement)
        return tokens

    def decode(self, token_idx):
        txt_list = [self.token[t] for t in token_idx]
        txt = ''.join(txt_list)
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        txt = txt.translate(SUB)
        return txt

    def encode(self, token):
        word = [self.token_idx[t] for t in token]
        return word

    def stepbystep_outputs(self, inp, task_fns):

        outputs = []
        cur_inp = inp

        for fn in task_fns:
            cur_inp = fn(cur_inp)
            outputs.append(cur_inp)

        return outputs

    def generate_task_token_document(self, split):
        """
        Generate a document of the form t, x, t(x)
        """
        token_idx = self.sample_token()
        space_idx = np.array([self.token_idx[' ']])
        start_idx = np.array([self.token_idx['S']])

        tasks = self.sample_task(split)
        task_idx = []
        for idx, ts in enumerate(tasks[0]):
            task_str = 'T' + str(idx) + '_' + str(ts)
            task_idx.append(self.token_idx[task_str])
        task_idx = np.array(task_idx)

        output = token_idx
        for ofn in tasks[2]:
            output = ofn(output)

        document = np.concatenate([
                    start_idx, task_idx, space_idx,
                    token_idx, space_idx, output])

        return document

    def generate_step_document(self, split):
        """
        Generate a document of the form t, x, t(x)
        """
        token_idx = self.sample_token()
        space_idx = np.array([self.token_idx[' ']])
        start_idx = np.array([self.token_idx['S']])

        tasks = self.sample_task(split)
        task_idx = []
        for idx, ts in enumerate(tasks[0]):
            if isinstance(ts, int):
                task_str = 'T' + str(idx) + '_' + str(ts)
            else:
                task_str = 'T' + str(ts[0]) + '_' + str(ts[1])
            task_idx.append(self.token_idx[task_str])
        task_idx = np.array(task_idx)

        outputs = self.stepbystep_outputs(token_idx, tasks[2])
        document = [start_idx, task_idx, space_idx, token_idx] 

        for out in outputs:
            document.append(space_idx)
            document.append(out)

        document = np.concatenate(document)

        return document

    def generate_document(self, split='train'):
        if not self.cfg.direct:
            return self.generate_step_document(split)
        else:
            return self.generate_task_token_document(split)

    def generate_corpus(self):
        corpus = []
        for i in tqdm.trange(self.cfg.ndocuments):
            corpus.append(self.generate_document())
        self.corpus = np.array(corpus)

        self.eval_corpus = {}
        for split in ['train', 'all']:
            corpus = []
            for i in tqdm.trange(self.cfg.neval_documents):
                corpus.append(self.generate_document(split))
            self.eval_corpus[split] = np.array(corpus)

        return self.corpus, self.eval_corpus

    def store_data(self):
        """
        Store the tokens into a file
        """
        # Store transition matrix, token dictionaries
        os.makedirs(self.fdir, exist_ok=True)

        pickle.dump(self.token_idx,
                    open(self.fdir + '/token_idx.pkl', 'wb'))
        pickle.dump(self.token,
                    open(self.fdir + '/token.pkl', 'wb'))

        np.save(self.fdir + '/corpus.npy', self.corpus)
        np.save(self.fdir + '/train_eval_corpus.npy',
                self.eval_corpus['train'])
        np.save(self.fdir + '/all_eval_corpus.npy',
                self.eval_corpus['all'])

        pickle.dump(self.functions_info,
                    open(self.fdir + '/functions_info.pkl', 'wb'))

        self.cfg = OmegaConf.to_container(self.cfg)
        json.dump(dict(self.cfg), open(self.fdir + '/config.json', 'w'),
                  indent=4)


class SyntheticDataset:
    """
    Dataset object to create a dataloader
    """
    def __init__(self, fpath, split='train'):
        datafiles = {
            "train": os.path.join(fpath, 'corpus.npy'),
            "train_eval": os.path.join(fpath, 'train_eval_corpus.npy'),
            "all_eval": os.path.join(fpath, 'all_eval_corpus.npy')
        }

        self.data = np.load(datafiles[split])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = torch.from_numpy(self.data[idx])
        dat, target = elem[:-1], elem[1:]
        return dat, target


class SyntheticEval(SyntheticData):
    """
    Create dataloader for each function composition
    """
    def __init__(self, net_cfg, nsamples, nbatch,
                 direct_eval=None, permute=None):
        # Evaluate step by step
        self.step_eval = not direct_eval

        # Use permute / bijections
        self.permute_eval = permute

        # Load data properties
        info_fname = os.path.join(net_cfg.data.path,
                                  "functions_info.pkl")
        data_fname = os.path.join(net_cfg.data.path,
                                  "config.json")

        self.token_idx = np.load(
            os.path.join(net_cfg.data.path, "token_idx.pkl"),
            allow_pickle=True)

        self.cfg = OmegaConf.create(json.load(open(data_fname)))
        self.special_tokens = [' ', '<PAD>', 'S']
        self.n_special = len(self.special_tokens)
        self.n_alphabets = self.cfg.n_alphabets

        # Create functions
        self.functions_info = np.load(info_fname, allow_pickle=True)
        self.composed_functions = self.functions_info['composition_reduced']

        # Network configuration
        self.net_cfg = net_cfg

        # Number of samples to evaluate 
        self.nsamples = nsamples
        # Number of functiosn to evaluate at same time
        self.nbatch = nbatch

        self.task_map()

    def get_seq_info(self, sample):
        """
        Properties of the sequence
        """
        seq_info = {}

        sp_idx = self.token_idx[' ']
        total_len = len(sample)

        if (not self.cfg['direct']) and self.cfg['task_tokens']:
            sp_pos = np.where(sample == sp_idx)[0]

            seq_info['last_space'] = sp_pos[-1]
            seq_info['prompt'] = sp_pos[1] + 1
            seq_info['new'] = total_len - seq_info['prompt']

        elif self.cfg['task_tokens']:
            sp_pos = np.where(sample == sp_idx)[0]

            seq_info['last_space'] = sp_pos[-1]
            seq_info['prompt'] = sp_pos[1] + 1
            seq_info['new'] = total_len - seq_info['prompt']
        else:
            raise ValueError

        return seq_info

    def generate_step_document(self, task_info):
        """
        Generate a document of the form t, x, t(x)
        """
        token_idx = self.sample_token()
        space_idx = np.array([self.token_idx[' ']])
        start_idx = np.array([self.token_idx['S']])

        task_idx = []
        for idx, ts in enumerate(task_info[0]):
            task_str = 'T' + str(idx) + '_' + str(ts)
            task_idx.append(self.token_idx[task_str])
        task_idx = np.array(task_idx)

        if self.step_eval:
            outputs = self.stepbystep_outputs(token_idx, task_info[2])
            document = [start_idx, task_idx, space_idx, token_idx]
            for out in outputs:
                document.append(space_idx)
                document.append(out)
        else:
            outputs = token_idx
            for fn in task_info[2]:
                outputs = fn(outputs)

            document = [start_idx, task_idx, space_idx, token_idx,
                        space_idx, outputs] 

        document = np.concatenate(document)

        return document

    @torch.no_grad()
    def evaluate_docs(self, net, dat, seq_info, device, lstm=False):

        shape = dat.shape
        if device == 'cuda':
            dat = dat.cuda(non_blocking=True)

        dat = dat.view(-1, shape[-1])
        inp_c = dat[:, :-1]
        inp = dat[:, :seq_info['prompt']]

        if lstm:
            net.hidden = None

        for _ in range(seq_info['new']):
            logits = net(inp)
            logits = logits[:, -1, :]
            inp_next = torch.argmax(logits, -1, keepdims=True)
            inp = torch.cat((inp, inp_next), dim=1)

        output = inp
        output_c = torch.argmax(net(inp_c), -1)

        output_l = output[:, seq_info['last_space']+1:]
        output_cl = output_c[:, seq_info['last_space']:]

        targets_l = dat[:, seq_info['last_space']+1:]

        # Accuracy averaged over all positions
        acc_l = (output_l.reshape(-1) == targets_l.reshape(-1))
        acc_l = acc_l.view(shape[0], shape[1], output_l.shape[-1])

        # Strict accuracy (1 if all tokens correct, 0 otherwise)
        acc_cl = (output_cl.reshape(-1) == targets_l.reshape(-1))
        acc_cl = acc_cl.view(shape[0], shape[1], output_cl.shape[-1])

        # Accuracy including the step-by-step tokens)
        total_acc = acc_l.float().mean((-1, -2)).to('cpu').numpy()
        sharp_acc = acc_l.all(-1).float().mean(-1).to('cpu').numpy()
        total_acc_c = acc_cl.float().mean((-1, -2)).to('cpu').numpy()

        return total_acc, sharp_acc, total_acc_c

    def get_acc(self, net, lstm=False):
        """
        Get the accuracy of each function composition and store seperately
        """
        info = self.functions_info

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        net.eval()
        if device == 'cuda':
            net = net.cuda()

        acc_map = {}

        tid_list = []
        doc_list = []

        if lstm:
            net.use_hidden = True

        for idx, tid in enumerate(info['task_id']):

            reduced_func = self.composed_functions[idx]
            task_funcs = []
            for d, t in enumerate(tid):

                if self.permute_eval and d == 0:
                    fnap = BaseFunction.permute
                else:
                    fnap = BaseFunction.map

                fn = functools.partial(
                    fnap,
                    mapping=info['functions'][d][t])
                task_funcs.append(fn)

            task_info = (tid, reduced_func, task_funcs)

            # Generate document for task composition
            docs = []
            for _ in range(self.nsamples):
                docs.append(self.generate_step_document(task_info))

            if idx == 0:
                sample = torch.Tensor(docs[0]).long()
                seq_info = self.get_seq_info(sample)

            tid_list.append(tid)
            doc_list.append(docs)

            if idx % self.nbatch == self.nbatch - 1 or \
                    idx == len(info['task_id']) - 1:

                flatten_docs = torch.Tensor(
                    np.array(doc_list, dtype=int)).long()

                acc_list = self.evaluate_docs(
                    net, flatten_docs, seq_info, device, lstm)

                for idx in range(len(tid_list)):
                    tid = tid_list[idx]
                    acc_map[tuple(tid)] = (acc_list[0][idx], acc_list[1][idx], acc_list[2][idx])

                tid_list = []
                doc_list = []

        return acc_map

    def save_accs(self, cfg, accs):
        self.fdir = 'data/' + cfg.tag
        os.makedirs(self.fdir, exist_ok=True)
        pickle.dump(accs,
                    open(self.fdir + '/accs.pkl', 'wb'))


class SyntheticEvalCombinatorial(SyntheticEval):
    """
    Evaluate on in-order and out-of-order functions
    """
    @torch.no_grad()
    def evaluate_docs(self, net, dat, seq_info, device):

        shape = dat.shape

        if device == 'cuda':
            dat = dat.cuda(non_blocking=True)

        dat = dat.view(-1, shape[-1])

        inp_c = dat[:, :-1]
        inp = dat[:, :seq_info['prompt']]

        for _ in range(seq_info['new']):
            logits = net(inp)
            logits = logits[:, -1, :]
            inp_next = torch.argmax(logits, -1, keepdims=True)
            inp = torch.cat((inp, inp_next), dim=1)

        output = inp
        output_c = torch.argmax(net(inp_c), -1)

        output_l = output[:, seq_info['last_space']+1:]
        output_cl = output_c[:, seq_info['last_space']:]

        targets_l = dat[:, seq_info['last_space']+1:]

        acc_l = (output_l.reshape(-1) == targets_l.reshape(-1))
        acc_l = acc_l.view(shape[0], shape[1], output_l.shape[-1])

        acc_cl = (output_cl.reshape(-1) == targets_l.reshape(-1))
        acc_cl = acc_cl.view(shape[0], shape[1], output_cl.shape[-1])

        total_acc = acc_l.float().mean((-2)).to('cpu').numpy()
        total_acc_c = acc_cl.float().mean((-2)).to('cpu').numpy()

        return total_acc, total_acc_c

    def generate_step_document(self, task_info):
        """
        Generate a document of the form t, x, t(x)
        """
        token_idx = self.sample_token()
        space_idx = np.array([self.token_idx[' ']])
        start_idx = np.array([self.token_idx['S']])

        task_idx = []
        for ts in task_info[0]:
            task_str = 'T' + str(ts[0]) + '_' + str(ts[1])
            task_idx.append(self.token_idx[task_str])
        task_idx = np.array(task_idx)

        outputs = self.stepbystep_outputs(token_idx, task_info[1])
        document = [start_idx, task_idx, space_idx, token_idx] 

        for out in outputs:
            document.append(space_idx)
            document.append(out)

        document = np.concatenate(document)

        return document

    def get_task_list(self, depth, choices):

        task_list = {}

        for num_identity in range(depth, -1, -1):
            num_funcs = depth - num_identity

            # Position of identity
            for id_pos in combinations(range(depth), num_identity):
                # Number of swapped positions
                for num_swap in range(num_funcs+1):
                    for sw_pos in combinations(range(num_funcs), num_swap):
                        
                        fix_pos = set(range(depth)) - set(id_pos)
                        fix_pos = tuple(fix_pos - set(sw_pos))

                        id_pos = tuple(id_pos)
                        sw_pos = tuple(sw_pos)

                        nfunc_choices = [None for d in range(depth)]

                        for pos in id_pos:
                            nfunc_choices[pos] = [(pos, 0)]

                        for pos in fix_pos:
                            nfunc_choices[pos] = [(pos, i) for i in range(1, choices)]

                        for pos in sw_pos:

                            nfunc_choices[pos] = []
                            for d in range(depth):
                                if pos != d:
                                    nfunc_choices[pos] += [(d, i) for i in range(1, choices)]

                        cur_tlist = list(itertools.product(*nfunc_choices))

                        # sample_num = min(len(cur_tlist), 1e9)
                        sample_num = min(len(cur_tlist), 500)

                        tlist = random.sample(cur_tlist, sample_num)
                        if (num_identity, num_swap) in task_list:
                            task_list[(num_identity, num_swap)] += tlist
                        else:
                            task_list[(num_identity, num_swap)] = tlist

        return task_list

    def get_acc(self, net):
        """
        Compute accuracies for compositions with
        - different number of identities (number of composition)
        - different displacements

        It is too expensive to compute all accuracies
        """


        depth = self.cfg.function.depth
        info = self.functions_info

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        net.eval()
        if device == 'cuda':
            net = net.cuda()

        acc_map = {}
        tid_list = []
        doc_list = []

        depth, choices = info['functions'].shape[0:2]

        task_list = self.get_task_list(depth, choices)

        for key in tqdm.tqdm(task_list):
            acc_map[key] = {}
            for idx, tsk in enumerate(task_list[key]):

                task_funcs = []
                for d, t in tsk:
                    fn = functools.partial(
                        BaseFunction.map,
                        mapping=info['functions'][d][t])
                    task_funcs.append(fn)

                task_info = (tsk, task_funcs)

                docs = []
                for _ in range(self.nsamples):
                    docs.append(self.generate_step_document(task_info))

                if idx == 0:
                    sample = torch.Tensor(docs[0]).long()
                    seq_info = self.get_seq_info(sample)

                tid_list.append(tsk)
                doc_list.append(docs)

                if idx % self.nbatch == self.nbatch - 1 or \
                        idx == len(task_list[key]) - 1:

                    flatten_docs = torch.Tensor(
                        np.array(doc_list, dtype=int)).long()

                    acc_list = self.evaluate_docs(
                        net, flatten_docs, seq_info, device)

                    for idx in range(len(tid_list)):
                        tid = tid_list[idx]

                        acc_map[key][tuple(tid)] = (acc_list[0][idx], acc_list[1][idx])

                    tid_list = []
                    doc_list = []

        return acc_map


def get_vocab_len(fpath):
    token = np.load(os.path.join(fpath, 'token.pkl'),
                    allow_pickle=True)
    return len(token)


def get_space_pos(fpath, loader):
    """
    Get positions of the space
    """
    token_idx = np.load(os.path.join(fpath, 'token_idx.pkl'),
                    allow_pickle=True)
    sp_idx = token_idx[' ']
    sp_pos = np.where(loader.dataset.data[0] == sp_idx)[0][-1]
    return sp_pos


def get_seq_info(fpath, loader):
    """
    Get markers in the seqeunce like length of prompt, last space
    """
    token_idx = np.load(os.path.join(fpath, 'token_idx.pkl'),
                    allow_pickle=True)
    seq_info = {}

    data_cfg = json.load(open(os.path.join(fpath, 'config.json')))

    sp_idx = token_idx[' ']
    sample = loader.dataset.data[0]
    total_len = len(sample)

    if (not data_cfg['direct']) and data_cfg['task_tokens']:
        sp_pos = np.where(loader.dataset.data[0] == sp_idx)[0]

        seq_info['last_space'] = sp_pos[-1]
        seq_info['prompt'] = sp_pos[1] + 1
        seq_info['new'] = total_len - seq_info['prompt']

    return seq_info


def get_trainLoader(cfg):
    dataset = SyntheticDataset(cfg.data.path, 'train')
    dataloader = DataLoader(dataset,
                            batch_size=cfg.data.batch_size,
                            shuffle=True, pin_memory=True,
                            num_workers=cfg.data.num_workers)
    return dataloader


def get_evalLoaders(cfg):
    """
    Create dataloaders for evaluation
    """
    loaders = []
    for split in ['train_eval', 'all_eval']:
        dataset = SyntheticDataset(cfg.data.path, split)
        loaders.append(DataLoader(
            dataset, batch_size=cfg.data.batch_size,
            shuffle=False, pin_memory=True,
            num_workers=cfg.data.num_workers))
    return loaders
