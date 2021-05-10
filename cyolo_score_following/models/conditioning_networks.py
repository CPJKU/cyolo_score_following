
import torch

import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from cyolo_score_following.models.custom_modules import Flatten, TemporalBatchNorm


class ContextConditioning(nn.Module):
    def __init__(self, zdim=128, n_lstm_layers=1, activation=nn.ELU, normalize_input=False,
                 spec_out=32, hidden_size=64, groupnorm=False):
        super(ContextConditioning, self).__init__()

        self.inplace = False

        if isinstance(activation, str):
            # if activation is provided as a string reference create a callable instance (if possible)
            activation = eval(activation)

        modules = []
        if normalize_input:
            print('Using input normalization!')
            modules.append(TemporalBatchNorm(78, affine=False))

        modules.extend([
            nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 24) if groupnorm else nn.BatchNorm2d(24),
            activation(self.inplace),
            nn.Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 24) if groupnorm else nn.BatchNorm2d(24),
            activation(self.inplace),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 48) if groupnorm else nn.BatchNorm2d(48),
            activation(self.inplace),
            nn.Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 48) if groupnorm else nn.BatchNorm2d(48),
            activation(self.inplace),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 96) if groupnorm else nn.BatchNorm2d(96),
            activation(self.inplace),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 96) if groupnorm else nn.BatchNorm2d(96),
            activation(self.inplace),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 96) if groupnorm else nn.BatchNorm2d(96),
            activation(self.inplace),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 96) if groupnorm else nn.BatchNorm2d(96),
            activation(self.inplace),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.GroupNorm(1, 96) if groupnorm else nn.BatchNorm2d(96),
            activation(self.inplace),
            Flatten(),
            nn.Linear(96 * 4 * 2, spec_out),
            nn.LayerNorm(spec_out) if groupnorm else nn.BatchNorm1d(spec_out),
            activation(self.inplace)])

        self.enc = nn.Sequential(*modules)

        self.kw, self.kh = 40, 78
        self.dw, self.dh = 1, 1

        self.seq_model = nn.LSTM(spec_out, hidden_size=hidden_size, num_layers=n_lstm_layers, batch_first=True)
        self.z_enc = nn.Sequential(
            nn.Linear(hidden_size + spec_out, zdim),
            nn.LayerNorm(zdim) if groupnorm else nn.BatchNorm1d(zdim),
            activation(self.inplace)
        )

        self.inference_x = deque(maxlen=self.kw)
        self.step_cnt = 0

    def forward(self, x, context):

        raise NotImplementedError

    def encode_sequence(self, x, hidden=None):

        x, last_steps, lengths = self.encode_samples(x)

        x = torch.nn.utils.rnn.pack_sequence(x, enforce_sorted=False)

        _, hidden = self.seq_model(x, hidden)

        # use hidden state of last layer as conditioning information z
        z = self.z_enc(torch.cat((hidden[0][-1], last_steps), -1))

        return z, hidden

    def encode_samples(self, x):

        last_steps = []

        zero_lengths = []
        for i in range(len(x)):

            if x[i].shape[0] < self.kw:
                padding = self.kw - x[i].shape[0]
                x[i] = F.pad(x[i], (0, 0, padding, 0), mode='constant')

            last_steps.append(x[i][-40:].unsqueeze(0))

            stacked = torch.stack(x[i][:self.kw * (x[i].shape[0]//self.kw)].split(self.kw)).unsqueeze(1)

            if stacked.shape[0] == 1:
                zero_lengths.append(i)
                # fill zero to make processing easier, will be overwritten later-on with zero in the output
                x[i] = torch.zeros_like(stacked)
            else:
                x[i] = stacked

        lengths = [spec.shape[0] for spec in x]
        last_steps = self.enc(torch.stack(last_steps))

        x = self.enc(torch.cat(x))

        x = list(torch.split(x, lengths))

        for idx in zero_lengths:
            x[idx] = torch.zeros(1, x[idx].shape[-1], device=x[idx].device)
            lengths[idx] = 1

        return x, last_steps, lengths

    def get_conditioning(self, x, hidden=None):

        self.inference_x.append(x)

        x = torch.cat(list(self.inference_x))

        if x.shape[0] < self.kw:
            padding = self.kw - x.shape[0]
            x = F.pad(x, (0, 0, padding, 0), mode='constant')

        last_steps = self.enc(x.unsqueeze(0).unsqueeze(0))

        if hidden is None:
            _, hidden = self.seq_model(torch.zeros_like(last_steps).unsqueeze(0), hidden)

        z = self.z_enc(torch.cat((hidden[0][-1], last_steps), -1))

        self.step_cnt += 1
        if self.step_cnt == self.kw:

            _, hidden = self.seq_model(last_steps.unsqueeze(0), hidden)
            self.step_cnt = 0

        return z, hidden
