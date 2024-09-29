from utils import download_dataset, create_dataloader, split_dataset
import hydra
from omegaconf import DictConfig, OmegaConf

from model import get_model, LitModel
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
# use the dataset folder to create dataloader for train, val and test
@hydra.main(version_base=None, config_path="configs", config_name="configs.yaml")
def main(cfg:DictConfig):
    print(OmegaConf.to_yaml(cfg))
    train_dataloader, val_dataloader, test_dataloader = create_dataloader(cfg)

    if cfg.mode == "train":
        lit_model = LitModel(cfg)
        # save the best val accuracy model
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_accuracy",
            mode="max",
            dirpath="checkpoints",
            filename="best-checkpoint",
        )
        trainer = L.Trainer(max_epochs=cfg.trainer.max_epochs, callbacks=[checkpoint_callback])
        trainer.fit(lit_model, train_dataloader, val_dataloader)
if __name__ == "__main__":
    
    main()
    
