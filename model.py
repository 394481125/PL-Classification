import timm
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.optim import AdamP
import torch.optim.lr_scheduler as lr_scheduelr
from torchmetrics.functional import accuracy,f1,auc,auroc
from config import *

class WCModel(pl.LightningModule):
    def __init__(self,optimizer,scheduler):
        super().__init__()
        model = timm.create_model('resnet50', pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(model.children())[:-3])
        self.head = WCHead(1024, 1024, 2)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        elif self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        elif self.optimizer == 'adamp':
            optimizer = AdamP(self.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-2)
        if self.scheduler == "reducelr":
            scheduler = lr_scheduelr.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode="max", verbose=True)
        elif self.scheduler == "cosineanneal":
            scheduler = lr_scheduelr.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5,
                                                                 last_epoch=-1, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/acc"}

    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
        image, label = image.cuda(), label.cuda()
        output = self(image)

        t_loss = self.criterion(output, label)
        t_acc = accuracy(output, label)
        t_f1 = f1(output, label)
        t_auroc = auroc(output, label,num_classes=NUMCLASS)

        self.log('train/loss', t_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log('train/t_acc', t_acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log('train/t_f1', t_f1, on_epoch=True, on_step=False, prog_bar=True)
        self.log('train/t_auroc', t_auroc, on_epoch=True, on_step=False, prog_bar=True)

        return  t_loss

    def validation_step(self, val_batch, batch_idx):
        image, label = val_batch
        image, label = image.cuda(), label.cuda()
        output = self(image)

        v_loss = self.criterion(output, label)
        v_acc = accuracy(output, label)
        v_f1 = f1(output, label)
        v_auroc = auroc(output, label,num_classes=NUMCLASS)

        self.log('val/v_loss', v_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val/v_acc', v_acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val/v_f1', v_f1, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val/v_auroc', v_auroc, on_epoch=True, on_step=False, prog_bar=True)

        return v_loss


class WCCNN(pl.LightningModule):
    def __init__(self, input_channel=3, output_channel=1, kernel_size=(3, 3), stride=(1, 1), padding=1,alpa=0.2):
        super(WCCNN, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.alpa = alpa
        self.iterN=5
        self.beta1=0.5
        self.beta2=1
        self.mu1=10
        self.mu2=10
        self.roph=1.1
        self.aux_var=1
        self.conv = nn.Conv2d(in_channels=self.input_channel, out_channels=self.output_channel, stride=self.stride,
                               padding=self.padding, kernel_size=self.kernel_size, bias=False)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.input_channel,out_channels=int(self.input_channel//16), kernel_size=1, stride=1, bias=False),
            nn.Conv2d(in_channels=int(self.input_channel//16),out_channels=self.input_channel, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def momn(self,x,att,iterN,beta1,beta2,mu1,mu2,roph,aux_var,mode='default'):
        with torch.no_grad():
            batchSize = x.size(0)
            dim = x.size(1)
            I1 = torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1)
            I3 = 3.0*I1
            normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
            A = x.div(normA.view(batchSize,1,1).expand_as(x))
            # initionlization
            J1 = A;J2 = A;Y = A;L1 = 0;L2 = 0;Z=torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,1,1)
            for i in range(0, iterN):
                if mode=='avg':
                    eta1=1.0/(2*mu1);eta2=1.0/(2*mu2);eta3=1.0/(2*mu1+2*mu2)
                else:
                    eta1=1.0/(mu1);eta2=1.0/(mu2);eta3=1.0/(mu1+mu2)
                if i==0:
                    # step1
                    J1 = Y-eta1*beta1*(I1*Y)
                    # step2
                    J2 = Y-eta2*beta2*(1-att)*Y
                    # step3
                    Y = Y+mu1*eta3*(J1-Y)+mu2*eta3*(J2-Y)
                    ZY = 0.5*(I3 - A)
                    Y = Y.bmm(ZY)
                    Z = ZY
                elif i==(iterN-1):
                    # step1
                    J1 = J1-eta1*mu1*(J1-Y)-eta1*L1
                    J1 = J1-beta1*eta1*(I1*J1)
                    # step2
                    J2 = J2-eta2*mu2*(J2-Y)-eta2*L2
                    J2 = J2-eta2*beta2*(1-att)*J2
                    # step3
                    Y = Y+mu1*eta3*(J1-Y)+mu2*eta3*(J2-Y)+eta3*(L1+L2)
                    ZY = 0.5*(I3 - Z.bmm(Y))
                    Y = Y.bmm(ZY)
                else:
                    # step1
                    J1 = J1-eta1*mu1*(J1-Y)-eta1*L1
                    J1 = J1-beta1*eta1*(I1*J1)
                    # step2
                    J2 = J2-eta2*mu2*(J2-Y)-eta2*L2
                    J2 = J2-eta2*beta2*(1-att)*J2
                    # step3
                    Y = Y+mu1*eta3*(J1-Y)+mu2*eta3*(J2-Y)+eta3*(L1+L2)
                    ZY = 0.5*(I3 - Z.bmm(Y))
                    Y = Y.bmm(ZY)
                    Z = ZY.bmm(Z)
                # step 4
                if(i<iterN-1):
                    L1 = aux_var*L1+mu1*(J1-Y)
                    L2 = aux_var*L2+mu2*(J2-Y)
                    mu1 = roph*mu1
                    mu2 = roph*mu2

            y = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        return y

    def triuvec(self,x):
        with torch.no_grad():
            batchSize = x.data.shape[0]
            dim = x.data.shape[1]
            dtype = x.dtype
            x = x.reshape(batchSize, dim * dim)
            I = torch.ones(dim, dim).triu().reshape(dim * dim)
            index = I.nonzero()
            y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(dtype)
            for i in range(batchSize):
                y[i, :] = x[i, index].t()
        return y

# 4,1,224,224
    def forward(self, input):
        batchSize = input.size(0)
        pre_output = self.conv(input)

        A_input = input.view(input.size(0), input.size(1), -1)
        A_input = A_input - torch.mean(A_input, dim=2, keepdim=True)
        A_input = 1. / A_input.size(2) * A_input.bmm(A_input.transpose(1, 2))

        S_attention = self.attention(input)
        S_attention = S_attention.view(S_attention.size(0),S_attention.size(1), -1)
        S_attention = S_attention.bmm(S_attention.transpose(1,2))

        with torch.no_grad():  # 这样不会生成计算图
            Y_output = self.momn(A_input,S_attention,self.iterN,
                                self.beta1,self.beta2,
                                self.mu1,self.mu2,self.roph,
                                self.aux_var)

        all_evals = 0
        for index in range(batchSize):
            evals, _ = torch.eig(Y_output[index], eigenvectors=True)
            all_evals += evals[:,0]
        avg_evals = all_evals/batchSize
        avg_evals = avg_evals.unsqueeze(1)
        avg_evals = avg_evals.unsqueeze(1)
        avg_evals = avg_evals.unsqueeze(0)
        avg_evals = torch.softmax(avg_evals,dim=1)


        pre_weight = self.conv.parameters()
        wc_weight = None
        for idx, param in enumerate(pre_weight):
            wc_weight = param + param * avg_evals

        output = F.conv2d(input, wc_weight, stride=self.stride, padding=self.padding)

        output = (pre_output+output)/2

        output = self.gap(output)

        return output

class WCHead(pl.LightningModule):
    def __init__(self, input_channel,hidden_channel,output_channel):
        super(WCHead, self).__init__()
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.output_channel = output_channel
        self.model = WCCNN(input_channel,hidden_channel)
        self.liner = nn.Linear(hidden_channel,output_channel)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        x = self.model(input)
        x = x.view(x.size(0),-1)
        x = self.dropout(x)
        output = self.liner(x)
        return output


if __name__ == '__main__':
    x = torch.randn(4, 3, 512, 512)
    model = WCModel('adam', 'cosineanneal')
    output = model(x)
    # model = WCHead(16,2048, 2048, 2)
    # output = model(output)
    # model = WCHead(64,256,6)
    # x = torch.randn(4,64,64,64)
    # y = model(x)
    # print(y.shape)