import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional
from model import EEGNet
from utils.pytorch_utils import init_weights, CenterLoss
from torch.utils.data import TensorDataset, DataLoader
from utils.ot import sinkhorn_loss_joint_IPOT


def FGSM(model: nn.Module,
         x: torch.Tensor,
         y: torch.Tensor,
         eps: Optional[float] = 0.05,
         distance: Optional[str] = 'inf',
         target: Optional[bool] = False):
    """ FGSM attack """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss().to(device)

    data_loader = DataLoader(dataset=TensorDataset(x, y),
                             batch_size=128,
                             shuffle=False,
                             num_workers=1,
                             drop_last=False)

    model.eval()
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.clone().detach().to(device)
        batch_y = batch_y.clone().detach().to(device)
        batch_x.requires_grad = True

        with torch.enable_grad():
            loss = criterion(model(batch_x), batch_y)
        grad = torch.autograd.grad(loss,
                                   batch_x,
                                   retain_graph=False,
                                   create_graph=False)[0]

        if distance == 'inf':
            delta = grad.detach().sign()
        elif distance == 'l2':
            grad_norms = torch.norm(
                grad.detach().view(len(batch_x), -1), p=2, dim=1) + 1e-10
            # factor = torch.min(eps / grad_norms, torch.ones_like(grad_norms))
            delta = grad.detach() / grad_norms.view(-1, 1, 1, 1)
        else:
            raise 'No such distance.'

        if target:
            batch_adv_x = batch_x.detach() - eps * delta
        else:
            batch_adv_x = batch_x.detach() + eps * delta

        if step == 0: adv_x = batch_adv_x
        else: adv_x = torch.cat([adv_x, batch_adv_x], dim=0)

    return adv_x


def PGD(model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        eps: Optional[float] = 0.05,
        alpha: Optional[float] = 0.005,
        steps: Optional[int] = 20,
        distance: Optional[str] = 'inf',
        target: Optional[bool] = False):
    """ PGD attack """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss().to(device)

    data_loader = DataLoader(dataset=TensorDataset(x, y),
                             batch_size=128,
                             shuffle=False,
                             drop_last=False)

    model.eval()
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.clone().detach().to(device)
        batch_y = batch_y.clone().detach().to(device)

        # craft adversarial examples
        batch_adv_x = batch_x.clone().detach() + torch.empty_like(
            batch_x).uniform_(-eps, eps)
        for _ in range(steps):
            batch_adv_x.requires_grad = True
            with torch.enable_grad():
                loss = criterion(model(batch_adv_x), batch_y)
            grad = torch.autograd.grad(loss,
                                       batch_adv_x,
                                       retain_graph=False,
                                       create_graph=False)[0]

            if distance == 'inf':
                delta = grad.detach().sign()
            elif distance == 'l2':
                grad_norms = torch.norm(
                    grad.detach().view(len(batch_x), -1), p=2, dim=1) + 1e-10
                delta = grad.detach() / grad_norms.view(-1, 1, 1, 1)
            else:
                raise 'No such distance'

            if target:
                batch_adv_x = batch_adv_x.detach() - alpha * delta
            else:
                batch_adv_x = batch_adv_x.detach() + alpha * delta

            # projection
            if distance == 'inf':
                delta = torch.clamp(batch_adv_x - batch_x, min=-eps, max=eps)
            else:
                delta = batch_adv_x - batch_x
                delta_norms = torch.norm(delta.view(len(batch_x), -1),
                                         p=2,
                                         dim=1)
                factor = torch.min(eps / delta_norms,
                                   torch.ones_like(delta_norms))
                delta = delta * factor.view(-1, 1, 1, 1)

            batch_adv_x = (batch_x + delta).detach()

        if step == 0: adv_x = batch_adv_x
        else: adv_x = torch.cat([adv_x, batch_adv_x], dim=0)

    return adv_x


def PGD_batch(model: nn.Module,
              x: torch.Tensor,
              y: torch.Tensor,
              eps: Optional[float] = 0.05,
              alpha: Optional[float] = 0.005,
              steps: Optional[int] = 20):
    """ PGD attack """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss().to(device)

    model.eval()
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)

    # craft adversarial examples
    adv_x = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    for _ in range(steps):
        adv_x.requires_grad = True
        with torch.enable_grad():
            loss = criterion(model(adv_x), y)
        grad = torch.autograd.grad(loss,
                                   adv_x,
                                   retain_graph=False,
                                   create_graph=False)[0]
        adv_x = adv_x.detach() + alpha * grad.detach().sign()
        # projection
        delta = torch.clamp(adv_x - x, min=-eps, max=eps)
        adv_x = (x + delta).detach()

    return adv_x


def maximize_shift_inconsistency(model: nn.Module,
              batch_adv_x: torch.Tensor,
              x: torch.Tensor,
              y: torch.Tensor,
              criterion: nn.Module,
              eps: Optional[float] = 0.05,
              alpha: Optional[float] = 0.005,
              steps: Optional[int] = 20):
    device = next(model.parameters()).device

    model.eval()
    batch_adv_x = batch_adv_x.clone().detach().to(device)
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)
    for _ in range(steps):
        batch_adv_x.requires_grad = True
        with torch.enable_grad():
            loss = criterion(model(batch_adv_x) - model(x), y)
        grad = torch.autograd.grad(loss,
                                batch_adv_x,
                                retain_graph=False,
                                create_graph=False)[0]
        batch_adv_x = batch_adv_x.detach() + 0.4 * alpha * grad.detach().sign()
        # projection
        delta = torch.clamp(batch_adv_x - x, min=-eps, max=eps)
        batch_adv_x = (x + delta).detach()
    
    return batch_adv_x


def feature_scatter(model: nn.Module,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    eps: Optional[float] = 0.05,
                    alpha: Optional[float] = 0.005,
                    steps: Optional[int] = 20):
    device = next(model.parameters()).device

    model.eval()
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)

    logits = model(x)
    m, n = len(x), len(x)

    # craft adversarial examples
    adv_x = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    for _ in range(steps):
        adv_x.requires_grad = True
        adv_logits = model(adv_x)
        with torch.enable_grad():
            loss = sinkhorn_loss_joint_IPOT(1, 0.00, logits,
                                                  adv_logits, None, None,
                                                  0.01, m, n)
        grad = torch.autograd.grad(loss,
                                   adv_x,
                                   retain_graph=False,
                                   create_graph=False)[0]
        adv_x = adv_x.detach() + alpha * grad.detach().sign()
        # projection
        delta = torch.clamp(adv_x - x, min=-eps, max=eps)
        adv_x = (x + delta).detach()
    
    return adv_x


def get_preds(model: nn.Module, x: torch.Tensor):
    logits = model(x)
    preds = nn.Softmax(dim=1)(logits).argmax(dim=1)
    return preds


def get_probs(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    logits = model(x)
    probs = torch.index_select(nn.Softmax(dim=1)(logits),
                               dim=1,
                               index=y.squeeze())
    return torch.diag(probs)


def SimBA(model: nn.Module,
          x: torch.Tensor,
          y: torch.Tensor,
          max_iters: Optional[float] = 0.5,
          eps=0.05,
          distance='inf',
          target=False):
    """ simple black attack """
    device = next(model.parameters()).device

    if distance == 'inf': alpha = eps
    else: alpha = 0.05

    data_loader = DataLoader(dataset=TensorDataset(x, y),
                             batch_size=128,
                             shuffle=False,
                             num_workers=1,
                             drop_last=False)

    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.clone().detach().to(device)
        batch_y = batch_y.clone().detach().to(device)

        shape = batch_x.shape
        n_dims = shape[-2] * shape[-1]
        iters = int(max_iters * n_dims)

        perm_idx_list = torch.randperm(n_dims)[:iters].to(device)
        perm = torch.zeros(shape[0], n_dims).to(device)
        queries = torch.zeros(shape[0]).to(device)
        remaining_idx = torch.arange(0, shape[0]).to(device)

        batch_x = batch_x.reshape(shape[0], -1)

        with torch.no_grad():
            preds = get_preds(model, batch_x.reshape(shape))
            probs = get_probs(model, batch_x.reshape(shape), batch_y)

            for i in tqdm(range(iters)):
                perm_idx = perm_idx_list[i]
                perm_x = batch_x[remaining_idx] + perm[remaining_idx]
                preds[remaining_idx] = get_preds(
                    model, perm_x.reshape((len(perm_x), *shape[1:])))

                if target:
                    remaining = ~preds.eq(batch_y.view_as(preds))
                else:
                    remaining = preds.eq(batch_y.view_as(preds))

                # if all inputs are misclassified
                if remaining.sum() == 0: break

                remaining_idx = torch.arange(0, shape[0]).to(device)
                remaining_idx = remaining_idx[remaining]
                one_step_perm = torch.zeros(remaining.sum(), n_dims).to(device)
                one_step_perm[:, perm_idx] = alpha
                # training negative direction
                perm[remaining_idx] -= one_step_perm
                perm_x = batch_x[remaining_idx] + perm[remaining_idx]
                new_probs = get_probs(
                    model,
                    perm_x.reshape((len(perm_x), *shape[1:])).to(device),
                    batch_y[remaining_idx])
                queries[remaining_idx] += 1
                effective = new_probs.lt(probs[remaining_idx])
                # perturb on positive direction if not effective
                if target:
                    perm[remaining_idx[
                        effective]] += 2 * one_step_perm[effective]
                    perm_x[effective] += 2 * one_step_perm[effective]
                else:
                    perm[remaining_idx[
                        ~effective]] += 2 * one_step_perm[~effective]
                    perm_x[~effective] += 2 * one_step_perm[~effective]
                # update probs
                probs[remaining_idx] = get_probs(
                    model,
                    perm_x.reshape((len(perm_x), *shape[1:])).to(device),
                    batch_y[remaining_idx])

        if distance == 'l2':
            perm_norms = torch.norm(perm.view(len(perm), -1), p=2, dim=1) + 1e-10
            perm = (perm / perm_norms.view(-1, 1)) * eps

        batch_adv_x = (batch_x + perm).reshape(shape)

        if step == 0: adv_x = batch_adv_x
        else: adv_x = torch.cat([adv_x, batch_adv_x], dim=0)

    return adv_x.cpu(), queries.cpu()


def get_pred(model: nn.Module, x: torch.Tensor):
    device = next(model.parameters()).device
    pred_x = torch.ones(size=(len(x), 1)).squeeze().type(torch.LongTensor)
    train_loader = DataLoader(dataset=TensorDataset(x, pred_x),
                              batch_size=32,
                              shuffle=False,
                              num_workers=1,
                              drop_last=False)

    idx = 0
    for batch_x, _ in train_loader:
        sub_pred = model(batch_x.to(device))
        sub_pred = nn.Softmax(dim=1)(sub_pred).cpu().argmax(dim=1)
        pred_x[idx:idx + len(sub_pred)] = sub_pred.type(torch.LongTensor)
        idx += len(sub_pred)

    return pred_x


def TrainSub(model: nn.Module, x_sub: torch.Tensor, y_sub: torch.Tensor,
             aug_repeat: int):
    device = next(model.parameters()).device
    sub_model = EEGNet(n_classes=len(np.unique(y_sub.numpy())),
                       Chans=x_sub.shape[2],
                       Samples=x_sub.shape[3],
                       kernLenght=64,
                       F1=4,
                       D=2,
                       F2=8,
                       dropoutRate=0.25).to(device)
    sub_model.apply(init_weights)

    params = [v for _, v in sub_model.named_parameters()]
    optimizer = optim.Adam(params, lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(device)

    # initial dataset
    model.eval()
    y_sub = get_pred(model, x_sub)

    for r in range(aug_repeat):
        train_loader = DataLoader(dataset=TensorDataset(x_sub, y_sub),
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=1,
                                  drop_last=False)
        for epoch in range(100):
            sub_model.train()
            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logit = sub_model(batch_x)
                loss = criterion(logit, batch_y)
                loss.backward()
                optimizer.step()
                sub_model.MaxNormConstraint()

        # 样本 augment
        if r < aug_repeat - 1:
            adv_x = FGSM(sub_model, x_sub, y_sub, eps=0.05)
            adv_y = get_pred(model, adv_x.cpu())
            x_sub = torch.cat([x_sub, adv_x.cpu()], dim=0)
            y_sub = torch.cat([y_sub, adv_y.type(torch.LongTensor)], dim=0)
            del adv_x, adv_y

    return sub_model


class RayS(object):
    def __init__(self, model, epsilon=0.031, order=np.inf):
        self.model = model
        self.ord = order
        self.epsilon = epsilon
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.queries = None
        self.device = next(model.parameters()).device

    def get_xadv(self, x, v, d, lb=0., ub=1.):
        if isinstance(d, int):
            d = torch.tensor(d).repeat(len(x)).to(self.device)
        out = x + d.view(len(x), 1, 1, 1) * v
        out = torch.clamp(out, lb, ub)
        return out

    def attack_hard_label(self, x, y, target=None, query_limit=10000, seed=None):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            (x, y): original image
        """
        shape = list(x.shape)
        dim = np.prod(shape[1:])
        lb, ub = x.min(), x.max()
        if seed is not None:
            np.random.seed(seed)

        # init variables
        self.queries = torch.zeros_like(y).to(self.device)
        self.sgn_t = torch.sign(torch.ones(shape)).to(self.device)
        self.d_t = torch.ones_like(y).float().fill_(float("Inf")).to(self.device)
        working_ind = (self.d_t > self.epsilon).nonzero().flatten()

        stop_queries = self.queries.clone()
        dist = self.d_t.clone()
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t, lb, ub)
 
        block_level = 0
        block_ind = 0
        for i in range(query_limit):
            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            valid_mask = (self.queries < query_limit) 
            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[valid_mask.nonzero().flatten(), start:end] *= -1.
            attempt = attempt.view(shape)

            self.binary_search(x, y, target, attempt, valid_mask)

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = torch.norm((self.x_final - x).view(shape[0], -1), self.ord, 1)
            stop_queries[working_ind] = self.queries[working_ind]
            working_ind = (dist > self.epsilon).nonzero().flatten()

            if torch.sum(self.queries >= query_limit) == shape[0]:
                print('out of queries')
                break

            print('d_t: %.4f | adbd: %.4f | queries: %.4f | rob acc: %.4f | iter: %d'
                   % (torch.mean(self.d_t), torch.mean(dist), torch.mean(self.queries.float()),
                      len(working_ind) / len(x), i + 1))
 

        stop_queries = torch.clamp(stop_queries, 0, query_limit)
        return self.x_final, stop_queries, dist, (dist <= self.epsilon)
    

    def attack_batch(self, x, y, target=None, query_limit=5000, seed=None):
        data_loader = DataLoader(dataset=TensorDataset(x, target if target!=None else y),
                             batch_size=1024,
                             shuffle=False,
                             drop_last=False)

        for step, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.clone().detach().to(self.device)
            batch_y = batch_y.clone().detach().to(self.device)

            batch_adv_x, _, _, _ = self.attack_hard_label(batch_x, batch_y, target=batch_y if target!=None else None, query_limit=query_limit)
            
            # projection
            if self.ord == np.inf:
                delta = torch.clamp(batch_adv_x - batch_x, min=-self.epsilon, max=self.epsilon)
            else:
                delta = batch_adv_x - batch_x
                delta_norms = torch.norm(delta.view(len(batch_x), -1),
                                         p=2,
                                         dim=1)
                factor = torch.min(self.epsilon / delta_norms,
                                   torch.ones_like(delta_norms))
                delta = delta * factor.view(-1, 1, 1, 1)

            batch_adv_x = (batch_x + delta).detach()

            if step == 0: adv_x = batch_adv_x
            else: adv_x = torch.cat([adv_x, batch_adv_x], dim=0)

        return adv_x.cpu()

    # check whether solution is found
    def search_succ(self, x, y, target, mask):
        self.queries[mask] += 1
        if target!=None:
            return self.model.predict_label(x[mask]) == target[mask]
        else:
            return self.model.predict_label(x[mask]) != y[mask]

    # binary search for decision boundary along sgn direction
    def binary_search(self, x, y, target, sgn, valid_mask, tol=1e-3):
        lb, ub = x.min(), x.max()
        sgn_norm = torch.norm(sgn.view(len(x), -1), 2, 1)
        sgn_unit = sgn / sgn_norm.view(len(x), 1, 1, 1)

        d_start = torch.zeros_like(y).float().to(self.device)
        d_end = self.d_t.clone()

        initial_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, self.d_t, lb, ub), y, target, valid_mask)
        to_search_ind = valid_mask.nonzero().flatten()[initial_succ_mask]
        d_end[to_search_ind] = torch.min(self.d_t, sgn_norm)[to_search_ind]

        while len(to_search_ind) > 0:
            d_mid = (d_start + d_end) / 2.0
            search_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, d_mid, lb, ub), y, target, to_search_ind)
            d_end[to_search_ind[search_succ_mask]] = d_mid[to_search_ind[search_succ_mask]]
            d_start[to_search_ind[~search_succ_mask]] = d_mid[to_search_ind[~search_succ_mask]]
            to_search_ind = to_search_ind[((d_end - d_start)[to_search_ind] > tol)]

        to_update_ind = (d_end < self.d_t).nonzero().flatten()
        if len(to_update_ind) > 0:
            self.d_t[to_update_ind] = d_end[to_update_ind]
            self.x_final[to_update_ind] = self.get_xadv(x, sgn_unit, d_end, lb, ub)[to_update_ind]
            self.sgn_t[to_update_ind] = sgn[to_update_ind]

    def __call__(self, data, label, target=None, query_limit=10000):
        return self.attack_hard_label(data, label, target=target, query_limit=query_limit)