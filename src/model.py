from typing import Dict, List, Optional
import pytorch_lightning as pl
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class Model(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 29,
        lr: Optional[float] = 1e-3,
        weight_decay: Optional[float] = 0,
        batch_size: Optional[int] = 64,
        optimizer: Optional[str] = None,
        loss = None,
        *args,
        **kwargs
    ) -> None:
        super(Model, self).__init__(*args, **kwargs)
        weights = ResNet50_Weights.DEFAULT
        original_model = resnet50(weights=weights)       
        # Everything except the last linear layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Linear(2048, num_classes)

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False
            
        self.lr = lr
        self.batch_size = batch_size
        self.loss = torch.nn.CrossEntropyLoss()

        if optimizer is None or optimizer == "Adam":
            self.optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=weight_decay
            )
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x: List[str]) -> List[str]:
        f = self.features(x)        
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

    def _inference_training(
        self, batch, batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        From https://huggingface.co/docs/transformers/model_doc/t5#training
        """
        img, target, prop, rect = batch
        output = self(prop)
        target_class = rect[:, 4].long()
        return self.loss(output, target_class)

    def training_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss = self._inference_training(batch, batch_idx)
        self.log("train loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss = self._inference_training(batch, batch_idx)
        self.log("val loss", loss, batch_size=self.batch_size, sync_dist=True)
        return loss

    def test_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss = self._inference_training(batch, batch_idx)
        self.log("test loss", loss, batch_size=self.batch_size, sync_dist=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer
