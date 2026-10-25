from dataclasses import dataclass


@dataclass
class StarConfig:
    vocab_size: int = 4096
    block_size: int = 256
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.1
    bias: bool = True
    flash: bool = True  # use F.scaled_dot_product_attention; False uses the explicit masked path
