from typing import Dict, List, Optional
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix
from src.loss import binary_focal_loss_with_logits
from torchvision.models import resnet50, ResNet50_Weights

class Model(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 29,
        lr: Optional[float] = 1e-3,
        weight_decay: Optional[float] = 0,
        batch_size: Optional[int] = 1,
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
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )
        self.modelName = 'LightCNN-29'
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
        if self.target_mask_supplied:
            data, target, mask = batch
        else:
            data, target = batch
        output = self(data)
        if self.target_mask_supplied:
            output *= mask[:,None,:,:]

        output, target = output, target
        #print(output.shape, target.shape)
        output = output[:,0,:,:]
        target = target.to(torch.float32)
        dice, iou, accuracy, sensitivity, specificity = self.metrics(output, target)

        #out_img = wandb.Image(
        #    output[0,...].cpu().detach().numpy().squeeze(), 
        #    caption="Prediction"
        #)
        #out_target = wandb.Image(
        #    target[0,...].cpu().detach().numpy().squeeze(), 
        #    caption="target"
        #)
        #self.logger.experiment.log({"prediction": [out_img, out_target]}) #, step = self.logger.experiment.current_trainer_global_step

        return self.loss(output, target), accuracy, specificity, iou, dice, sensitivity

    def training_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss, accuracy, specificity, iou, dice, sensitivity = self._inference_training(batch, batch_idx)
        self.log("train loss", loss, batch_size=self.batch_size)
        self.log("train accuracy", accuracy, batch_size=self.batch_size)
        self.log("train specificity", specificity, batch_size=self.batch_size)
        self.log("train iou", iou, batch_size=self.batch_size)
        return loss

    def validation_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss, accuracy, specificity, iou, dice, sensitivity = self._inference_training(batch, batch_idx)
        self.log("val loss", loss, batch_size=self.batch_size, sync_dist=True)
        self.log("val accuracy", accuracy, batch_size=self.batch_size, sync_dist=True)
        self.log("val specificity", specificity, batch_size=self.batch_size, sync_dist=True)
        self.log("val iou", iou, batch_size=self.batch_size, sync_dist=True)
        return loss

    def test_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss, accuracy, specificity, iou, dice, sensitivity = self._inference_training(batch, batch_idx)
        self.log("test loss", loss, batch_size=self.batch_size, sync_dist=True)
        self.log("test accuracy", accuracy, batch_size=self.batch_size, sync_dist=True)
        self.log("test specificity", specificity, batch_size=self.batch_size, sync_dist=True)
        self.log("test iou", iou, batch_size=self.batch_size, sync_dist=True)
        self.log("test dice", dice, batch_size=self.batch_size, sync_dist=True)
        self.log("test sensitivity", sensitivity, batch_size=self.batch_size, sync_dist=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer

    def metrics(self, preds, target):
        # Dice
        X = target.view(-1)
        Y = torch.sigmoid(preds.view(-1)) > 0.5

        Y = Y*1.0
        dice = 2*torch.mean(torch.mul(X,Y))/torch.mean(X+Y)

        # Intersection over Union
        IoU = torch.mean(torch.mul(X,Y))/(torch.mean(X+Y)-torch.mean(torch.mul(X,Y)))

        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(X, Y).ravel()
        accuracy = (tp+tn)/(tp+tn+fp+fn)		
        
        # Sensitivity
        sensitivity = tp/(tp+fn)

        # Specificity
        specificity = tn/(tn+fp)

        return dice, IoU, accuracy, sensitivity, specificity