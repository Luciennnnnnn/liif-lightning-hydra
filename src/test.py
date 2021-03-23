# lightning imports
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import seed_everything

# hydra imports
from omegaconf import DictConfig
from hydra.utils import log
import hydra

# normal imports
from typing import List, Optional

# src imports
from src.utils import template_utils
from .models.liif import LIIF


def test(config: DictConfig) -> Optional[float]:
    """Example of inference with trained model.
    It loads trained image classification model from checkpoint.
    Then it loads example image and predicts its label.
    """

    # load model from checkpoint
    # model __init__ parameters will be loaded from ckpt automatically
    # you can also pass some parameter explicitly to override it

    # load Lightning model
    log.info(f"Loading model from<{config.CKPT_PATH}>")
    trained_model = LIIF.load_from_checkpoint(checkpoint_path=config.CKPT_PATH)

    # print model hyperparameters
    print(trained_model.hparams)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    
    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger,  _convert_="partial"
    )

    log.info("Starting testing!")
    trainer.test(model=trained_model, datamodule=datamodule)