# Compositional Capabilities of Autoregressive Transformers: A Study on Synthetic, Interpretable Tasks

Code for our ICML 24 paper:  [Compositional Capabilities of Autoregressive Transformers: A Study on Synthetic, Interpretable Tasks][https://arxiv.org/abs/2311.12997]

**Summary.** We create a synthetic setup to evaluate the ability of autoregressive Transformers to learn function compositions. We find that: (1) Autoregressive Transformers learn function compositions using very compositions in the training data (unlike LSTMs); (2) generating intermediate outputs when composing functions is more effective for generalizing to new, unseen compositions; (3) the attention layers select which function to apply while the feed-forward layers execute the selected capability. 

## Setup

We use [micromamba](https://mamba.readthedocs.io/en/latest/installation.html) as the package manager. To install the packages run:

```
micromamba create -y -f env.yml
micromamba activate composition
```

# Usage

**Step 1**: Generate training data using `01_generate_data.py`. The config file `config/gen/conf.yaml` can be modified to generate prompts in the direct or step-by-step formats. The config file also controls other choices like the number of in-order or out-of-order compositions. 

**Step 2**: Train model using `02_train.py`. Modify `config/train/conf.yaml` to use the data generated in step 1.

**Step 3**: Evaluate data on in-order (`03_evaluate_i.py`) or out-of-order (`03_evaluate_o.py`) compositions. Note that during evaluation, the model must autoregressively generate the outputs. Modify 


```bash
python 01_generate_data.py
python 02_train.py
python 03_evaluate_i.py
```

The default config runs all 3 steps in less than 10 minutes.

## Directory structure

├── 01_generate_data.py. # Generate train data
├── 02_train.py                   # Train networks
├── 03_evaluate_i.py         # Evaluating in-order functions
├── 03_evaluate_o.py.       # Evaluating out-of-order functions
├── env.yml                          # Environment files 
├── config/                           # Config files
├── net/                                 # Training scripts and architectures
│   ├── lstm.py
│   ├── nanogpt.py               
│   └── runner.py                   # Training scripts for Transformer
├── run.sh
└── synthetic
    ├── functions.py.    # Create functions and compositions
    ├── generator.py.    # Generate prompts for training and eval
    └── init.py                # Load config and set random seed
