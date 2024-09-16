import torch
from torch_geometric.nn.models import GAE
from torch import Tensor
from typing import Optional, Tuple

class MyGAE(GAE):
    """
    The DTGN model, a Graph Autoencoder (GAE) for gene expression data.

    Parameters:
    encoder (Module): The encoder neural network module.
    decoder (Module): The decoder neural network module.

    Methods:
    total_loss(feat, z, edges, alpha=1):
        Computes the total loss as a combination of reconstruction and feature loss.


    Example:
    >>> encoder = YourEncoder()
    >>> decoder = YourDecoder()
    >>> model = MyGAE(encoder, decoder)
    >>> loss = model.total_loss(feat, z, edges)
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, None)
        self.feat_decoder = decoder

    # Calculate the feature loss
    def feat_loss(self, one_hot_pos, de_feat):
        feat_loss_func = torch.nn.CrossEntropyLoss()
        nums_class = de_feat.shape[-1]
        de_feat = de_feat.reshape((-1,nums_class))
        return feat_loss_func(de_feat,one_hot_pos)

    # Calculate the reconstruction loss
    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:

        z = z.permute(1, 0, 2).reshape((z.size(1), -1))
        return super().recon_loss(z, pos_edge_index, neg_edge_index)

    # the total loss
    def total_loss(self, feat, z, edges, alpha=1):
        return self.recon_loss(z, edges) + self.feat_loss(feat, z, edges) * alpha

    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        z = z.permute(1, 0, 2).reshape((z.size(1), -1))
        return super().test(z, pos_edge_index, neg_edge_index)