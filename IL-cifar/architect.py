""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch


class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_X, trn_y, xi, w_optim, net_weights, flag='cifar10'):
        """
        Compute unrolled weight w' and h` (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
            net_weights: net_weights: interleaved weights
        """
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y, flag) # L_trn(alpha, w, h)

        # interleaving_weight_decay = 1  # [1, 0.1, 0.01, 0.001, 0.0001]

        for name, param in self.net.named_weights(flag=flag):
            if net_weights is not None and name in net_weights:
                loss += 0.5 * self.w_weight_decay * torch.pow((param - net_weights[name]).norm(2), 2)
                # loss += 0.5 * interleaving_weight_decay * torch.pow((param - net1_weights[name]).norm(2), 2)
            else:
                loss += 0.5 * self.w_weight_decay * torch.pow(param.norm(2), 2)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights(flag))

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(flag), self.v_net.weights(flag), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim, net_weights, flag='cifar10'):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
            net_weights: net_weights: interleaved weights
        """
        # do virtual step (calc w` and h`)
        self.virtual_step(trn_X, trn_y, xi, w_optim, net_weights, flag)

        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y, flag) # L_val(alpha, w`, h`)

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_arch_weights = tuple(self.v_net.arch_weights(flag))
        v_nonarch_weights = tuple(self.v_net.nonarch_weights(flag))
        v_grads = torch.autograd.grad(loss, v_alphas + v_arch_weights + v_nonarch_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas): len(v_alphas) + len(v_arch_weights)]
        dh = v_grads[len(v_alphas) + len(v_arch_weights):]

        hessian = self.compute_hessian(dw, dh, trn_X, trn_y, flag)

        # update final gradient = dalpha - xi*hessian
        # with torch.no_grad():
        #     for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
        #         alpha.grad = da - xi*h
        
        # accumulate the graduate += dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                if alpha.grad is None:
                    alpha.grad = da - xi * h
                else:
                    alpha.grad.add_(da - xi * h)

    def compute_hessian(self, dw, dh, trn_X, trn_y, flag='cifar10'):
        """
        dw = dw` { L_val(alpha, w`, h`) }, dh = dh` { L_val(alpha, w`, h`) }
        w+ = w + eps_w * dw, h+ = h + eps_h * dh
        w- = w - eps_w * dw, h- = h - eps_h * dh
        hessian_w = (dalpha { L_trn(alpha, w+, h) } - dalpha { L_trn(alpha, w-, h) }) / (2*eps_w)
        hessian_h = (dalpha { L_trn(alpha, w, h+) } - dalpha { L_trn(alpha, w, h-) }) / (2*eps_h)
        eps_w = 0.01 / ||dw||, eps_h = 0.01 / ||dh||
        """
        norm_w = torch.cat([w.view(-1) for w in dw]).norm()
        eps_w = 0.01 / norm_w
        norm_h = torch.cat([h.view(-1) for h in dh]).norm()
        eps_h = 0.01 / norm_h

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.arch_weights(flag), dw):
                p += eps_w * d
        loss = self.net.loss(trn_X, trn_y, flag)
        dalpha_wpos = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(alpha, w+, h) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.arch_weights(flag), dw):
                p -= 2. * eps_w * d
        loss = self.net.loss(trn_X, trn_y, flag)
        dalpha_wneg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(alpha, w-, h) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.arch_weights(flag), dw):
                p += eps_w * d

        hessian_w = [(p-n) / (2.*eps_w) for p, n in zip(dalpha_wpos, dalpha_wneg)]

        # h+ = hw + eps*dh`
        with torch.no_grad():
            for p, d in zip(self.net.nonarch_weights(flag), dh):
                p += eps_h * d
        loss = self.net.loss(trn_X, trn_y, flag)
        dalpha_hpos = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(alpha, w, h+) }

        # h- = h - eps*dh`
        with torch.no_grad():
            for p, d in zip(self.net.nonarch_weights(flag), dh):
                p -= 2. * eps_h * d
        loss = self.net.loss(trn_X, trn_y, flag)
        dalpha_hneg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(alpha, w, h-) }

        # recover h
        with torch.no_grad():
            for p, d in zip(self.net.nonarch_weights(flag), dh):
                p += eps_h * d

        hessian_h = [(p-n) / (2.*eps_h) for p, n in zip(dalpha_hpos, dalpha_hneg)]

        # hessian
        hessian = [hw + hh for hw, hh in zip(hessian_w, hessian_h)]

        return hessian
