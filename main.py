from utils import download_dataset, create_dataloader, split_dataset
import hydra
from omegaconf import DictConfig, OmegaConf

# use the dataset folder to create dataloader for train, val and test
@hydra.main(version_base=None, config_path="configs", config_name="configs.yaml")
def main(cfg:DictConfig):
    print(OmegaConf.to_yaml(cfg))
    train_dataloader, val_dataloader, test_dataloader = create_dataloader(cfg)

if __name__ == "__main__":
    main()
    
