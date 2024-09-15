# @Author : CyIce
# @Time : 2024/6/25 10:19

import torch


def one_hot_encode(feat, num_intervals, origin_val=False):
    """
    Applies one-hot encoding to represent gene expression data.

    Parameters:
    feat (torch.Tensor): A tensor containing the gene expression data with shape (T, N, 1).
     num_intervals (int): The number of intervals for encoding.
    origin_val (bool, optional): If True, use the original values for encoding; otherwise, use binary encoding. Defaults to False.

    Returns:
    tuple: A tuple containing:
        - one_hot_feat (torch.Tensor): The one-hot encoded feature tensor with shape (stage_nums, n, num_intervals).
        - one_hot_pos (torch.Tensor): A tensor of interval indices with shape (stage_nums * n).

    Example:
    >>> feat = torch.tensor([[[0.1], [0.5]], [[0.2], [0.8]]])
    >>> one_hot_feat, one_hot_pos = one_hot_encode(feat, 5)
    """

    max_value = torch.max(feat)
    min_value = torch.min(feat)
    stage_nums = feat.shape[0]
    n = feat.shape[1]
    interval_width = (max_value - min_value) / num_intervals
    one_hot_feat = torch.zeros((stage_nums, n, num_intervals))
    one_hot_pos = torch.zeros((stage_nums, n, 1), dtype=torch.long)
    for i in range(stage_nums):
        for j in range(n):
            interval_index = min(num_intervals - 1, int((feat[i, j, 0] - min_value) // interval_width))
            if origin_val:
                one_hot_feat[i, j, interval_index] = feat[i, j, 0]
                one_hot_pos[i, j, 0] = interval_index
            else:
                one_hot_feat[i, j, interval_index] = 1
                one_hot_pos[i, j, 0] = interval_index
    one_hot_pos = one_hot_pos.reshape((-1))
    return one_hot_feat, one_hot_pos.reshape((-1))
