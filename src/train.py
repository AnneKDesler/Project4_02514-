import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from src.load_data import get_dataloaders_WASTE, get_dataloaders_proposals
from src.model import Model

def train(config=None, checkpoint_callbacks=None):
    with wandb.init(config=config, 
                    project="project4_02514",
                    entity="chrillebon",):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        lr = wandb.config.lr
        weight_decay = wandb.config.weight_decay
        epochs = wandb.config.epochs
        batch_size = wandb.config.batch_size

        device = 0
        
        model = Model(
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size
        )

        wandb.watch(model, log=None, log_freq=1)
        logger = pl.loggers.WandbLogger(project="project4_02514", entity="chrillebon")

        path = "/u/data/s194333/DLCV/Project4_02514-/data"
        trainloader, valloader,_ = get_dataloaders_proposals(batch_size=batch_size, num_workers=8, data_path=path, proposal_path = 'region_proposals6')

        # make sure no models are saved if no checkpoints are given
        if checkpoint_callbacks is None:
            checkpoint_callbacks = [
                ModelCheckpoint(monitor=False, save_last=False, save_top_k=0)
            ]

        trainer = pl.Trainer(
            max_epochs=epochs,
            default_root_dir="",
            callbacks=checkpoint_callbacks,
            accelerator="gpu",
            devices=[device],
            logger=logger,
            log_every_n_steps=1,
        )

        trainer.fit(
            model=model,
            train_dataloaders=trainloader,
            val_dataloaders=valloader,
        )

        print("Done!")

if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(dirpath="models/lr0.0001wd0.001", filename="best")
    train(
        config="src/config/default_params.yaml",
        checkpoint_callbacks=[checkpoint_callback],
    )
