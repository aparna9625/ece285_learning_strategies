""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.search_cells import SearchCell
import genotypes as gt
import logging
from torch.nn.parallel._functions import Broadcast
from collections import OrderedDict

def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies

class SearchCNN(nn.Module):
    """ Search CNN model """
    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers//3, 2*n_layers//3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits
    
    def arch_weights(self):
        _arch_weights = []
        for n, p in self.named_parameters():
            if 'linaer' not in n:
                _arch_weights.append((n, p))
        return _arch_weights
    
    def nonarch_weights(self):
        _nonarch_weights = []
        for n, p in self.named_parameters():
            if 'linear' in n:
                _nonarch_weights.append((n, p))
        return _nonarch_weights


class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, C1_in, C2_in, C, n1_classes, n2_classes, n_layers, criterion, n_nodes=4, stem_multiplier=3,
                 device_ids=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(i+2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3 * torch.randn(i+2, n_ops)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net1 = SearchCNN(C1_in, C, n1_classes, n_layers, n_nodes, stem_multiplier)
        self.net2 = SearchCNN(C2_in, C, n2_classes, n_layers, n_nodes, stem_multiplier)

    def forward(self, x, flag='cifar10'):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        logits = None
        if flag == 'cifar10':
            logits = self.net1(x, weights_normal, weights_reduce)
        elif flag == 'cifar100':
            logits = self.net2(x, weights_normal, weights_reduce)
        else:
            raise ValueError(flag)

        return logits

        # scatter x
        # xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        # wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        # wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

        # replicate modules
        # if flag == 'cifar10':
        #     replicas = nn.parallel.replicate(self.net1, self.device_ids)
        #     outputs = nn.parallel.parallel_apply(replicas,
        #                                         list(zip(xs, wnormal_copies, wreduce_copies)),
        #                                         devices=self.device_ids)
        #     return nn.parallel.gather(outputs, self.device_ids[0])
        # else:
        #     replicas = nn.parallel.replicate(self.net2, self.device_ids)
        #     outputs = nn.parallel.parallel_apply(replicas,
        #                                         list(zip(xs, wnormal_copies, wreduce_copies)),
        #                                         devices=self.device_ids)
        #     return nn.parallel.gather(outputs, self.device_ids[0])


    def loss(self, X, y, flag='cifar10'):
        logits = self.forward(X, flag)
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]
        
        gene_normal = gt.parse(weights_normal, k=2)
        gene_reduce = gt.parse(weights_reduce, k=2)
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self, flag='cifar10'):
        params = None
        if flag == 'cifar10':
            params = self.net1.parameters()
        elif flag == 'cifar100':
            params = self.net2.parameters()
        else:
            raise ValueError(flag)
        return params

    def named_weights(self, flag='cifar10'):
        named_params = None
        if flag == 'cifar10':
            named_params = self.net1.named_parameters()
        elif flag == 'cifar100':
            named_params = self.net2.named_parameters()
        else:
            raise ValueError(flag)
        return named_params
    
    def arch_weights(self, flag='cifar10'):
        if flag == 'cifar10':
            for name, param in self.net1.arch_weights():
                yield param
        elif flag == 'cifar100':
            for name, param in self.net2.arch_weights():
                yield param
        else:
            raise ValueError(flag)
    
    def nonarch_weights(self, flag='cifar10'):
        if flag == 'cifar10':
            for name, param in self.net1.nonarch_weights():
                yield param
        elif flag == 'cifar100':
            for name, param in self.net2.nonarch_weights():
                yield param
        else:
            raise ValueError(flag)

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
    
    def save_arch_weights(self, path, flag):
        state_dict = None
        if flag == 'cifar10':
            state_dict = self.net1.state_dict()
        elif flag == 'cifar100':
            state_dict = self.net2.state_dict()
        else:
            raise ValueError(flag)
        
        weight_dict = OrderedDict()
        for param in state_dict:
            if 'linear' not in param:
                weight_dict[param] = state_dict[param]
        torch.save(weight_dict, path)
    
    def load_arch_weights(self, state_dict, path):
        weights_dict = torch.load(path)
        state_dict.update(weights_dict)
        return state_dict, weights_dict
