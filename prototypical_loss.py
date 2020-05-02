# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


def euclidean(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def cosine(x, y):
    '''
    Compute consine similarity between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    cos = torch.nn.CosineSimilarity(dim=2,eps=1e-6)

    return -cos(x,y)


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support, dist_func, reg):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support
        if dist_func == "cosine":
            self.dist_func = cosine
        elif dist_func == "euclidean":
            self.dist_func = euclidean
        else:
            self.dist_func = None
        self.reg = reg

    def forward(self, input, target, weights):
        return prototypical_loss(input, target, self.n_support, weights=weights, dist_func=self.dist_func, lambda_reg=self.reg)


def prototypical_loss(input, target, n_support, weights, dist_func=euclidean, lambda_reg=0.05):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appertaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)  # non-repeated classes (i.e. types of ground truth)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])  # 每一个class的类似centroid?
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = dist_func(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    # loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    # --------------------------
    reg = 0
    for param in weights:
        param = param.to('cpu')
        reg += torch.sum(0.5*(param**2))  # L2 regularization
        # reg += torch.sum(torch.abs(param))  # L1 regularization
        
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean() + lambda_reg*reg
    # --------------------------

    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val
