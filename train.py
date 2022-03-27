from pytorch_lightning import Trainer
from Models.LitModel import LitModel
from DataModule.MNISTDataModule import MNISTDataModule

# import timm
# from pprint import pprint
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)



dm = MNISTDataModule()
model = LitModel()
trainer = Trainer()
trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm)
trainer.validate(datamodule=dm)
trainer.predict(datamodule=dm)