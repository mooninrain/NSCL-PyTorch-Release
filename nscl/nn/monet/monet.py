"""C. P. Burgess et al., "MONet: Unsupervised Scene Decomposition and Representation," pp. 1–22, 2019."""
from itertools import chain

import torch
from torch import nn

from . import networks



class MONet(nn.Module):
    def __init__(self):
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
            import pdb; pdb.set_trace()

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
            {'monet/m':torch.cat([getattr(self,'m{}'.format(k)) for k in range(self.num_slots)],dim=1), #[batch_size,num_slots,h_m,w_m]
            'monet/x':torch.cat([getattr(self,'x{}'.format(k)) for k in range(self.num_slots)],dim=1), #[batch_size,num_slots,h_m,w_m]
            'monet/xm':torch.cat([getattr(self,'xm{}'.format(k)) for k in range(self.num_slots)],dim=1), #[batch_size,num_slots,h_m,w_m]
            'monet/x_input':getattr(self,'x'), 'monet/x_tilde':getattr(self,'x_tilde')})

# class MONetModel(BaseModel):
#     def __init__(self, opt):
#         """Initialize this model class.

#         Parameters:
#             opt -- training/test options

#         A few things can be done here.
#         - (required) call the initialization function of BaseModel
#         - define loss function, visualization images, model names, and optimizers
#         """
#         BaseModel.__init__(self, opt)  # call the initialization method of BaseModel

#         self.beta = 0.5
#         self.gamma = 0.5
#         self.num_slots = 11


#         self.loss_names = ['E', 'D', 'mask']
#         self.visual_names = ['m{}'.format(i) for i in range(self.num_slots)] + \
#                             ['x{}'.format(i) for i in range(self.num_slots)] + \
#                             ['xm{}'.format(i) for i in range(self.num_slots)] + \
#                             ['x', 'x_tilde']
#         self.model_names = ['Attn', 'CVAE']
#         self.netAttn = networks.init_net(networks.Attention(3, 1), gpu_ids=self.gpu_ids)
#         self.netCVAE = networks.init_net(networks.ComponentVAE(3, 16), gpu_ids=self.gpu_ids)
#         self.eps = torch.finfo(torch.float).eps
#         # define networks; you can use opt.isTrain to specify different behaviors for training and test.
#         if self.isTrain:  # only defined during training time
#             self.criterionKL = nn.KLDivLoss(reduction='batchmean')
#             self.optimizer = optim.RMSprop(chain(self.netAttn.parameters(), self.netCVAE.parameters()), lr=opt.lr)
#             self.optimizers = [self.optimizer]

#     def set_input(self, input):
#         """Unpack input data from the dataloader and perform necessary pre-processing steps.

#         Parameters:
#             input: a dictionary that contains the data itself and its metadata information.
#         """
#         self.x = input['A'].to(self.device)
#         self.image_paths = input['A_paths']

#     def forward(self):
#         """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
#         self.loss_E = 0
#         self.x_tilde = 0
#         b = []
#         m = []
#         m_tilde_logits = []

#         # Initial s_k = 1: shape = (N, 1, H, W)
#         shape = list(self.x.shape)
#         shape[1] = 1
#         log_s_k = self.x.new_zeros(shape)

#         for k in range(self.num_slots):
#             # Derive mask from current scope
#             if k != self.num_slots - 1:
#                 log_alpha_k = self.netAttn(self.x, log_s_k)
#                 log_m_k = log_s_k + log_alpha_k
#                 # Compute next scope
#                 log_s_k += (1. - log_alpha_k.exp()).clamp(min=self.eps).log()
#             else:
#                 log_m_k = log_s_k

#             # Get component and mask reconstruction, as well as the z_k parameters
#             m_tilde_k_logits, x_mu_k, x_logvar_k, z_mu_k, z_logvar_k = self.netCVAE(self.x, log_m_k, k == 0)

#             # KLD is additive for independent distributions
#             self.loss_E += -0.5 * (1 + z_logvar_k - z_mu_k.pow(2) - z_logvar_k.exp()).sum()

#             m_k = log_m_k.exp()
#             x_k_masked = m_k * x_mu_k

#             # Exponents for the decoder loss
#             b_k = log_m_k - 0.5 * x_logvar_k - (self.x - x_mu_k).pow(2) / (2 * x_logvar_k.exp())
#             b.append(b_k.unsqueeze(1))

#             # Get outputs for kth step
#             setattr(self, 'm{}'.format(k), m_k * 2. - 1.) # shift mask from [0, 1] to [-1, 1]
#             setattr(self, 'x{}'.format(k), x_mu_k)
#             setattr(self, 'xm{}'.format(k), x_k_masked)

#             # Iteratively reconstruct the output image
#             self.x_tilde += x_k_masked
#             # Accumulate
#             m.append(m_k)
#             m_tilde_logits.append(m_tilde_k_logits)

#         self.b = torch.cat(b, dim=1)
#         self.m = torch.cat(m, dim=1)
#         self.m_tilde_logits = torch.cat(m_tilde_logits, dim=1)

#         return self.m

#     def backward(self):
#         """Calculate losses, gradients, and update network weights; called in every training iteration"""
#         n = self.x.shape[0]
#         self.loss_E /= n
#         self.loss_D = -torch.logsumexp(self.b, dim=1).sum() / n
#         self.loss_mask = self.criterionKL(self.m_tilde_logits.log_softmax(dim=1), self.m)
#         loss = self.loss_D + self.beta * self.loss_E + self.gamma * self.loss_mask

#         return loss