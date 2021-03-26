import hydra
from hydra.utils import log
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Learn more here: https://github.com/facebookresearch/hydra/issues/934
    import dotenv
    import torch
    from src.train import train
    from src.test import test
    from src.utils import template_utils

    # load environment variables from `.env` file if it exists
    dotenv.load_dotenv(dotenv_path=".env", override=True)

    # A couple of optional utilities:
    # - disabling python warnings
    # - disabling lightning logs
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    template_utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        log.info(f"Pretty printing config with Rich! <{config.print_config=}>")
        template_utils.print_config(config, resolve=True)
    # Train model
    if config.get("train"):
        return train(config)
    else:
        return test(config)


if __name__ == "__main__":
    main()
