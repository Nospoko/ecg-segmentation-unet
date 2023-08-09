import torch
import torch.nn as nn

from models.modules.attention_layers import MultiHeadAttention
from models.modules.conv_layers import PreNorm, Residual, Upsample, Downsample, ResnetBlock


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        dim_mults: tuple[int] = (1, 2, 4, 8),
        kernel_size: int = 3,
        resnet_block_groups: int = 4,
    ):
        super().__init__()

        # list of dimensions
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        # get list of corresponding ins and outs
        self.ins_outs = list(zip(dims[:-1], dims[1:]))
        self.num_resolutions = len(self.ins_outs)
        self.kernel_size = kernel_size
        self.resnet_block_groups = resnet_block_groups

        # initial conv
        self.init_conv = nn.Conv1d(in_channels, dim, kernel_size=1)

        # down blocks
        self.down = self._build_down_architecture()

        # middle blocks
        mid_dim = dims[-1]
        self.mid_block_1 = ResnetBlock(mid_dim, mid_dim, kernel_size=kernel_size)
        self.mid_attn = Residual(PreNorm(mid_dim, MultiHeadAttention(mid_dim)))
        self.mid_block_2 = ResnetBlock(mid_dim, mid_dim, kernel_size=kernel_size)

        # up blocks
        self.up = self._build_up_architecture()

        # final blocks
        self.final_resnet_block = ResnetBlock(dim * 2, dim, kernel_size=kernel_size, groups=self.resnet_block_groups)
        self.final_conv = nn.Conv1d(dim, out_channels, kernel_size=1)

    def _build_down_architecture(self) -> nn.ModuleList:
        down = nn.ModuleList([])

        for idx, (ins, outs) in enumerate(self.ins_outs):
            is_last = idx >= (self.num_resolutions - 1)

            down.append(
                nn.ModuleList(
                    [
                        ResnetBlock(ins, ins, kernel_size=self.kernel_size, groups=self.resnet_block_groups),
                        ResnetBlock(ins, ins, kernel_size=self.kernel_size, groups=self.resnet_block_groups),
                        Residual(PreNorm(ins, MultiHeadAttention(ins))),
                        Downsample(ins, outs) if not is_last else nn.Conv1d(ins, outs, kernel_size=3, padding=1),
                    ]
                )
            )

        return down

    def _build_up_architecture(self) -> nn.ModuleList:
        up = nn.ModuleList([])

        for idx, (ins, outs) in enumerate(reversed(self.ins_outs)):
            is_last = idx == (self.num_resolutions - 1)

            up.append(
                nn.ModuleList(
                    [
                        ResnetBlock(outs + ins, outs, kernel_size=self.kernel_size, groups=self.resnet_block_groups),
                        ResnetBlock(outs + ins, outs, kernel_size=self.kernel_size, groups=self.resnet_block_groups),
                        Residual(PreNorm(outs, MultiHeadAttention(outs))),
                        Upsample(outs, ins) if not is_last else nn.Conv1d(outs, ins, kernel_size=3, padding=1),
                    ]
                )
            )

        return up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # initial
        x = self.init_conv(x)
        r = x.clone()

        # list for storing residual connections in UNet
        residual_connections = []

        # down
        for block_1, block_2, attn, downsample in self.down:
            x = block_1(x)
            residual_connections.append(x)

            x = block_2(x)
            x = attn(x)
            residual_connections.append(x)

            x = downsample(x)

        # middle
        x = self.mid_block_1(x)
        x = self.mid_attn(x)
        x = self.mid_block_2(x)

        # up
        for block_1, block_2, attn, upsample in self.up:
            x = torch.cat([x, residual_connections.pop()], dim=1)
            x = block_1(x)

            x = torch.cat([x, residual_connections.pop()], dim=1)
            x = block_2(x)
            x = attn(x)

            x = upsample(x)

        # final
        x = torch.cat([x, r], dim=1)

        x = self.final_resnet_block(x)

        x = self.final_conv(x)

        return x
