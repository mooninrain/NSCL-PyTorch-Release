"""C. P. Burgess et al., "MONet: Unsupervised Scene Decomposition and Representation," pp. 1–22, 2019."""
from itertools import chain

import torch
from torch import nn

from . import networks



class MONet(nn.Module):
    def __init__(self,pretrained_monet):
        super().__init__()

        self.beta = 0.5
        self.gamma = 0.5
        self.num_slots = 11

        self.loss_names = ['E', 'D', 'mask']
        self.visual_names = ['m{}'.format(i) for i in range(self.num_slots)] + \
                            ['x{}'.format(i) for i in range(self.num_slots)] + \
                            ['xm{}'.format(i) for i in range(self.num_slots)] + \
                            ['x', 'x_tilde']
        self.model_names = ['Attn', 'CVAE']
        self.netAttn = networks.init_net(networks.Attention(3, 1))
        self.netCVAE = networks.init_net(networks.ComponentVAE(3, 16))
        self.eps = torch.finfo(torch.float).eps

        self.criterionKL = nn.KLDivLoss(reduction='batchmean')

    def forward(self,x_input):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        # x_input [batch_size, n_channels=3, h, w]

        self.x = x_input
        self.loss_E = 0
        self.x_tilde = 0
        b = []
        m = []
        m_tilde_logits = []

        # Initial s_k = 1: shape = (N, 1, H, W)
        shape = list(x_input.shape)
        shape[1] = 1
        log_s_k = x_input.new_zeros(shape)

        for k in range(self.num_slots):
            # Derive mask from current scope
            if k != self.num_slots - 1:
                log_alpha_k = self.netAttn(x_input, log_s_k)
                log_m_k = log_s_k + log_alpha_k
                # Compute next scope
                log_s_k += (1. - log_alpha_k.exp()).clamp(min=self.eps).log()
            else:
                log_m_k = log_s_k

            # Get component and mask reconstruction, as well as the z_k parameters
            m_tilde_k_logits, x_mu_k, x_logvar_k, z_mu_k, z_logvar_k = self.netCVAE(x_input, log_m_k, k == 0)

            # KLD is additive for independent distributions
            self.loss_E += -0.5 * (1 + z_logvar_k - z_mu_k.pow(2) - z_logvar_k.exp()).sum()

            m_k = log_m_k.exp()
            x_k_masked = m_k * x_mu_k

            # Exponents for the decoder loss
            b_k = log_m_k - 0.5 * x_logvar_k - (x_input - x_mu_k).pow(2) / (2 * x_logvar_k.exp())
            b.append(b_k.unsqueeze(1))

            # Get outputs for kth step
            setattr(self, 'm{}'.format(k), m_k * 2. - 1.) # shift mask from [0, 1] to [-1, 1]
            setattr(self, 'x{}'.format(k), x_mu_k)
            setattr(self, 'xm{}'.format(k), x_k_masked)

            # Iteratively reconstruct the output image
            self.x_tilde += x_k_masked
            # Accumulate
            m.append(m_k)
            m_tilde_logits.append(m_tilde_k_logits)

        self.b = torch.cat(b, dim=1)
        self.m = torch.cat(m, dim=1)
        self.m_tilde_logits = torch.cat(m_tilde_logits, dim=1)

        return self.m

    def get_monitor(self):
        n = self.x.shape[0]
        self.loss_E /= n
        self.loss_D = -torch.logsumexp(self.b, dim=1).sum() / n
        self.loss_mask = self.criterionKL(self.m_tilde_logits.log_softmax(dim=1), self.m)
        loss = self.loss_D + self.beta * self.loss_E + self.gamma * self.loss_mask

        return ({'loss/monet':loss,'loss/monet_D':self.loss_D,'loss/monet_E':self.loss_E,'loss/monet_mask':self.loss_mask},
            {'monet/m':torch.stack([getattr(self,'m{}'.format(k)) for k in range(self.num_slots)],dim=1), #[batch_size,num_slots,n_channel=1,h_m,w_m]
            'monet/x':torch.stack([getattr(self,'x{}'.format(k)) for k in range(self.num_slots)],dim=1), #[batch_size,num_slots,n_channel=3,h_m,w_m]
            'monet/xm':torch.stack([getattr(self,'xm{}'.format(k)) for k in range(self.num_slots)],dim=1), #[batch_size,num_slots,n_channel=3,h_m,w_m]
            'monet/x_input':getattr(self,'x'), 'monet/x_tilde':getattr(self,'x_tilde')}) #[batch_size,n_channel=3,h_m,w_m]

    def load_networks(self,pretrained_monet):
        for net_name in ['Attn','CVAE']:
            load_path = os.path.join(pretrained_monet,'latest_net_%s.pth'%net_name)
            net = getattr(self, 'net'+net_name)
            state_dict = torch.load(load_path, map_location=str(net.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            for key in list(state_dict.keys()):
                self.__patch_instance_norm_state_dict(state_dict,net,key.split('.'))
            net.load_state_dict(state_dict)
            
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)