"""Paper-faithful recurrent generator and discriminator modules."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import RCGANConfig


class RecurrentGenerator(nn.Module):
    """Separate noise/context LSTMs followed by the paper's condition skip."""

    def __init__(
        self,
        *,
        condition_size: int,
        output_size: int,
        config: RCGANConfig,
    ) -> None:
        super().__init__()
        self.noise_recurrent = nn.LSTM(
            input_size=config.latent_size,
            hidden_size=config.noise_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.context_recurrent = nn.LSTM(
            input_size=condition_size,
            hidden_size=config.context_hidden_size,
            num_layers=config.context_layers,
            batch_first=True,
        )
        combined_size = (
            config.noise_hidden_size + config.context_hidden_size + condition_size
        )
        self.dense = nn.Linear(combined_size, config.dense_hidden_size)
        self.activation = nn.LeakyReLU(config.leaky_relu_slope)
        self.output = nn.Linear(config.dense_hidden_size, output_size)

    def forward(self, noise: Tensor, conditions: Tensor) -> Tensor:
        noise_features, _ = self.noise_recurrent(noise)
        context_features, _ = self.context_recurrent(conditions)
        combined = torch.cat((noise_features, context_features, conditions), dim=-1)
        return self.output(self.activation(self.dense(combined)))


class RecurrentDiscriminator(nn.Module):
    """Framewise recurrent real/fake logits with condition skip connection.

    The target-availability mask is a LEEM adaptation.  Real and generated target
    values are masked identically, and the discriminator receives the mask so that
    missing look-ahead stations cannot be mistaken for physical zero errors.
    """

    def __init__(
        self,
        *,
        condition_size: int,
        target_size: int,
        config: RCGANConfig,
    ) -> None:
        super().__init__()
        recurrent_input_size = target_size + target_size + condition_size
        self.recurrent = nn.LSTM(
            input_size=recurrent_input_size,
            hidden_size=config.discriminator_hidden_size,
            num_layers=config.discriminator_layers,
            batch_first=True,
            dropout=0.0,
        )
        skip_size = config.discriminator_hidden_size + condition_size
        self.dense = nn.Linear(skip_size, config.dense_hidden_size)
        self.activation = nn.LeakyReLU(config.leaky_relu_slope)
        self.dropout = nn.Dropout(config.discriminator_dropout)
        self.output = nn.Linear(config.dense_hidden_size, 1)

    def forward(
        self,
        targets: Tensor,
        conditions: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        recurrent_input = torch.cat(
            (targets * target_mask, target_mask, conditions), dim=-1
        )
        recurrent_features, _ = self.recurrent(recurrent_input)
        combined = torch.cat((recurrent_features, conditions), dim=-1)
        hidden = self.dropout(self.activation(self.dense(combined)))
        return self.output(hidden).squeeze(-1)


def initialize_paper_weights(module: nn.Module) -> None:
    """Apply the initialization scheme reported by Arnelid et al."""

    for child in module.modules():
        if isinstance(child, nn.LSTM):
            for name, parameter in child.named_parameters(recurse=False):
                if "weight" in name:
                    nn.init.trunc_normal_(parameter, mean=0.0, std=0.1, a=-0.2, b=0.2)
                else:
                    nn.init.zeros_(parameter)
        elif isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
