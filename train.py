import pytorch_lightning as pl
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from flash.image import ImageClassificationData, ImageClassifier
from torch.utils.data.dataloader import DataLoader
import timm
from model import WCModel
from config import *




if __name__ == '__main__':
    pl.seed_everything(SEED)
    transform = T.Compose(
        [T.Resize([512, 512]),
         T.transforms.Grayscale(num_output_channels=1),
         T.ToTensor(),
         T.Normalize(mean=(0.5,),
                     std=(0.5,))])
    dataset = datasets.ImageFolder(DATASETS_PATH, transform=transform)
    train_length = int(0.8 * len(dataset))
    test_length = len(dataset) - train_length
    train_dataset, test_dataset = random_split(dataset, [train_length, test_length])

    test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=NUMWORKER)

    ksplits = KFold(NUMFOLD).split(range(len(train_dataset)))
    for idx, (train_index, val_index) in enumerate(ksplits):
        checkpoint_callback = ModelCheckpoint(
            monitor="val/v_acc",
            dirpath="saved",
            filename=f"{MODELNAME}_fold{idx + 1:02d}_" + "{val/v_acc:.4f}",
            save_top_k=3,
            mode="max",
            # save_weights_only=True
        )
        early_stop_callback = EarlyStopping(monitor="val/v_loss", min_delta=0.00, patience=10, verbose=True,
                                            mode="min")


        model = WCModel('adamp','cosineanneal')

        train_fold = Subset(train_dataset, train_index)
        val_fold = Subset(train_dataset, val_index)
        train_dataloader = DataLoader(train_fold,batch_size=BATCHSIZE,num_workers=NUMWORKER,shuffle=SHUFFLE)
        val_dataloader = DataLoader(val_fold,batch_size=BATCHSIZE,num_workers=NUMWORKER,shuffle=SHUFFLE)

        trainer = pl.Trainer(gpus=GPUS,
                             precision=16,
                             max_epochs=1,
                             log_every_n_steps=10,
                             strategy='ddp',
                             stochastic_weight_avg=True,
                             callbacks=[checkpoint_callback, early_stop_callback],
                             )

        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)



