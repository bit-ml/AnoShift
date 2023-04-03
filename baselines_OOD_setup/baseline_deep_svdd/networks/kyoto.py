import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

IN_SIZE = 571
REP_DIM = 50

class Kyoto_Net(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = REP_DIM
        
        self.encoder = nn.Sequential(
            torch.nn.Linear(IN_SIZE, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
#             torch.nn.ReLU(),
#             torch.nn.Linear(364, 400),
#             torch.nn.ReLU(),
#             torch.nn.Linear(400, 500)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Kyoto_Net_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()
        self.rep_dim = REP_DIM
        
        self.encoder = nn.Sequential(
            torch.nn.Linear(IN_SIZE, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
#             torch.nn.ReLU(),
#             torch.nn.Linear(364, 400),
#             torch.nn.ReLU(),
#             torch.nn.Linear(400, 500)
        )
          
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = nn.Sequential(
#             torch.nn.Linear(500, 400),
#             torch.nn.ReLU(),
#             torch.nn.Linear(400, 364),
#             torch.nn.ReLU(),
            torch.nn.Linear(50, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, IN_SIZE),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        print("x", x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)

        return x
