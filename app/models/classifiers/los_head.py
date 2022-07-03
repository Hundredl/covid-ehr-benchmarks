from torch import nn
import torch

class LosHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(LosHead, self).__init__()
        self.hidden_dim = (hidden_dim,)
        self.output_dim = (output_dim,)
        self.act = act_layer()
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(drop)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() == True else "cpu")

    def forward(self, x):
        # x = self.act(x)
        x = self.drop(x.to(device=self.device))
        x = self.fc(x)
        x = self.drop(x)
        return x
