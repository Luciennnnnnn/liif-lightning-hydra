<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?logo=pytorch"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-blue"></a>
[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/hobogalaxy/lightning-hydra-template)

</div>

## Description
An reimplement of [Learning Continuous Image Representation with Local Implicit Image Function
](https://arxiv.org/abs/2012.09161) using lightning and hydra.

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

Train model with default configuration
```yaml
python run.py
```

Train model with chosen experiment configuration
```yaml
# experiment configurations are placed in folder `configs/experiment/`
python run.py +experiment=exp_example_simple
```

You can override any parameter from command line like this
```yaml
python run.py trainer.max_epochs=20 optimizer.lr=0.0005
```

Train on GPU
```yaml
python run.py trainer.gpus=1
```
<br>
