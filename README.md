<div align="center">

# LIIF with Lightning + Hydra

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?logo=pytorch"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-blue"></a>
[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/hobogalaxy/lightning-hydra-template)

</div>

## Description
An reimplement of [Learning Continuous Image Representation with Local Implicit Image Function
](https://arxiv.org/abs/2012.09161) using lightning and hydra based on [this](https://github.com/hobogalaxy/lightning-hydra-template) awesome template.

## How to run
Install dependencies
```yaml
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda env create -f conda_env_gpu.yaml -n your_env_name
conda activate your_env_name

# install requirements
pip install -r requirements.txt
```

Train model with default configuration (train on DIV2K with visible GPUs)
```yaml
python run.py

# specify used GPUs
python run.py trainer.gpus=[0, 2, 5]

# use cpu
python run.py trainer.gpus=0
```

Train model with chosen experiment configuration
```yaml
# experiment configurations are placed in folder `configs/experiment/`
python run.py +experiment=exp_example_simple

# train on CelebAHQ
python run.py +experiment=train_on_CelebAHQ_32_256
python run.py +experiment=train_on_CelebAHQ_64_128
```

Test model
```yaml
# test pretrained model on DIV2K
python run.py train=false
```

You can override any parameter from command line like this
```yaml
python run.py trainer.max_epochs=20 optimizer.lr=0.0005
```
<br>
