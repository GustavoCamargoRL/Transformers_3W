from torch.nn import Module, ModuleList
import torch
from module.encoder import Encoder
import math
import torch.nn.functional as F


class Transformer(Module):
    def __init__(self,
                 d_model: int,
                 d_input: int,
                 d_channel: int,
                 d_output: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 device: str,
                 dropout: float = 0.1,
                 pe: bool = False,
                 mask: bool = False):
        """
        Custom Transformer model with step-wise and channel-wise encoders,
        followed by a gating mechanism to combine both representations.

        Args:
            d_model: Dimension of the embedding space.
            d_input: Input sequence length (time steps).
            d_channel: Number of channels/features per input step.
            d_output: Dimension of the model output.
            d_hidden: Hidden layer size in the encoder feed-forward network.
            q: Query dimension for multi-head attention.
            v: Value dimension for multi-head attention.
            h: Number of attention heads.
            N: Number of encoder layers (for both step-wise and channel-wise).
            device: Device to run computations on ('cpu' or 'cuda').
            dropout: Dropout probability.
            pe: Whether to add positional encoding to step-wise embeddings.
            mask: Whether to apply attention masking (typically for training).
        """
        super(Transformer, self).__init__()

        # Step-wise encoder stack
        self.encoder_list_1 = ModuleList([
            Encoder(d_model=d_model,
                    d_hidden=d_hidden,
                    q=q,
                    v=v,
                    h=h,
                    mask=mask,
                    dropout=dropout,
                    device=device) for _ in range(N)
        ])

        # Channel-wise encoder stack
        self.encoder_list_2 = ModuleList([
            Encoder(d_model=d_model,
                    d_hidden=d_hidden,
                    q=q,
                    v=v,
                    h=h,
                    dropout=dropout,
                    device=device) for _ in range(N)
        ])

        # Linear layers for embeddings
        self.embedding_channel = torch.nn.Linear(d_channel, d_model)  # step-wise embedding
        self.embedding_input = torch.nn.Linear(d_input, d_model)      # channel-wise embedding

        # Gate mechanism (decides weight between step-wise & channel-wise)
        self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel, 2)

        # Final output projection
        self.output_linear = torch.nn.Linear(d_model * d_input + d_model * d_channel, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model

    def forward(self, x, stage):
        """
        Forward pass through the Transformer.

        Args:
            x: Input tensor of shape [batch, time, channel].
            stage: String flag indicating training or testing stage.
                   (Masking is applied only during training if enabled.)

        Returns:
            output: Final model output (after gate & projection).
            encoding: Combined step-wise + channel-wise representation.
            score_input: Attention score matrix from step-wise encoder.
            score_channel: Attention score matrix from channel-wise encoder.
            input_to_gather: Step-wise embedding before encoders.
            channel_to_gather: Channel-wise embedding before encoders.
            gate: Softmax gate values (weights for combining both encoders).
        """
        # ----- Step-wise encoding -----
        encoding_1 = self.embedding_channel(x)
        input_to_gather = encoding_1

        # Optional positional encoding
        if self.pe:
            pe = torch.ones_like(encoding_1[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape: [input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)
            encoding_1 = encoding_1 + pe

        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage)

        # ----- Channel-wise encoding -----
        encoding_2 = self.embedding_input(x.transpose(-1, -2))
        channel_to_gather = encoding_2

        for encoder in self.encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage)

        # Flatten [batch, time, d_model] → [batch, time*d_model]
        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

        # ----- Gate mechanism -----
        gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
        encoding = torch.cat([
            encoding_1 * gate[:, 0:1],
            encoding_2 * gate[:, 1:2]
        ], dim=-1)

        # ----- Final projection -----
        output = self.output_linear(encoding)

        return output, encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate
