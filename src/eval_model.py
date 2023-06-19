import argparse
import pytorch_lightning as pl
import torch
import os
from src.load_data import get_dataloaders_proposals
from src.load_data import get_dataloaders_WASTE
from src.model import Model, DilatedNet


def eval(model_src):
    if not os.path.isfile(model_src):
        model_src = os.path.join("models", model_src)

    model = Model.load_from_checkpoint(checkpoint_path=model_src)

    #trainloader, valloader, testloader = get_dataloaders_DRIVE(batch_size=8, data_path="data/DRIVE/training")
    _, _, testloader = get_dataloaders_proposals(batch_size=1, data_path="data", proposal_path="region_proposals")

    model.to("cuda")

    predictions = dict()

    for img, target, prop, rect in testloader:
        output = model(prop.to("cuda"))
        output = torch.softmax(output)
        output = output.detach().cpu().numpy()
        output = output.squeeze()
        pred = out.argmax(axis=0)
        img_id = rect[5]
        if img_id not in predictions:
            predictions[str(img_id)] = dict()
            predictions[str(img_id)]["pred"] = [rect.append(pred)]
        else:
            predictions[str(img_id)]["pred"].append(rect.append(pred))





        break

    if torch.cuda.is_available():
        trainer = pl.Trainer(
            default_root_dir="",
            accelerator="gpu",
            devices=[0]
        )
    else:
        trainer = pl.Trainer(default_root_dir="")
    
    results = trainer.test(model=model, dataloaders=trainloader, verbose=True)

    print(results)

    results = trainer.test(model=model, dataloaders=valloader, verbose=True)

    print(results)

    results = trainer.test(model=model, dataloaders=testloader, verbose=True)

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="best.ckpt",
        type=str,
        help="path to ckpt file to evaluate",
    )

    args = parser.parse_args()

    eval(args.path)
