import torch
import torch.nn as nn
import torch.nn.functional as F


class AggregationTP(nn.Module):
    def __init__(self, feat_dim, *args, **kwargs):
        super().__init__()
        print("instantiating " + self.__class__.__name__)

        self.feat_dim = feat_dim

    def forward(self, x, b, t):
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        x = x.permute(0, 2, 1)
        f = F.avg_pool1d(x, t)
        f = f.view(b, self.feat_dim)
        return f


class AggregationTA(nn.Module):
    def __init__(self, feat_dim, *args, **kwargs):
        super().__init__()
        print("instantiating " + self.__class__.__name__)

        self.feat_dim = feat_dim
        self.middle_dim = 256
        self.attention_conv = nn.Conv2d(
            self.feat_dim, self.middle_dim, [8, 4]
        )  # 8, 4 cooresponds to 256, 128 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)

    def forward(self, x, b, t):

        # spatial attention
        a = F.relu(self.attention_conv(x))

        # arrange into batch temporal view
        a = a.view(b, t, self.middle_dim)

        # temporal attention
        a = a.permute(0, 2, 1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        a = F.softmax(a, dim=1)

        # global avg pooling of conv features
        x = F.avg_pool2d(x, x.size()[2:])

        # apply temporal attention
        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)
        f = att_x.view(b, self.feat_dim)

        return f


class AggregationRNN(nn.Module):
    def __init__(self, feat_dim, *args, **kwargs):
        super().__init__()
        print("instantiating " + self.__class__.__name__)

        self.hidden_dim = 512
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.feat_dim = self.hidden_dim

    def forward(self, x, b, t):
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)

        # apply LSTM
        output, (h_n, c_n) = self.lstm(x)
        output = output.permute(0, 2, 1)
        f = F.avg_pool1d(output, t)
        f = f.view(b, self.hidden_dim)

        return f


AGGREGATION = {
    "tp": AggregationTP,
    "ta": AggregationTA,
    "rnn": AggregationRNN,
}
