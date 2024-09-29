from timm import create_model
import torch.nn.functional as F
import torch
import lightning as L

def get_model(cfg):
    model = create_model(cfg.model.name, pretrained=cfg.model.pretrained, num_classes=cfg.model.num_classes)
    return model

class LitModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = get_model(cfg)
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('train_accuracy', accuracy, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_accuracy', accuracy, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optimizer.lr)
        return optimizer