import torch as t
import torch.nn as nn
import numpy as np
from collections import OrderedDict
mse = nn.MSELoss(reduction='none')
import torch.nn.functional as F

# self.fs = [self.data_dims[0], self.data_dims[0] // 2, self.data_dims[0] // 4, self.data_dims[0] // 8,
#            self.data_dims[0] // 16]
def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)

class reshape(nn.Module):
    def __init__(self, *args):
        super(reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class fc_relu(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.lin = nn.Linear(input, output)

    def forward(self, x):
        return F.leaky_relu(self.lin(x))

class conv2d_t_relu(nn.Module):
    def __init__(self, input, output, upsample):
        super().__init__()
        if upsample:
            self.convT = nn.ConvTranspose2d(input, output, 4, 2, 1)
        else:
            self.convT = nn.ConvTranspose2d(input, output, 3, 1, 1)

    def forward(self, x):
        return F.leaky_relu(self.convT(x))

class conv2d_relu(nn.Module):
    def __init__(self, input, output, downsample):
        super().__init__()
        if downsample:
            self.convT = nn.Conv2d(input, output, 4, 2, 1, bias=False)
        else:
            self.convT = nn.Conv2d(input, output, 3, 1, 1, bias=False)

    def forward(self, x):
        return F.leaky_relu(self.convT(x))


class energy_model(nn.Module):
    def __init__(self, z_dim, num_layers, ndf):
        super().__init__()
        self.z_dim = z_dim
        current_dims = self.z_dim
        layers = OrderedDict()
        for i in range(num_layers):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, ndf)
            layers['lrelu{}'.format(i+1)] = nn.LeakyReLU(0.2)
            current_dims = ndf

        layers['out'] = nn.Linear(current_dims, 1)
        self.energy = nn.Sequential(layers)

    def forward(self, zs):
        z = t.cat(zs, dim=1)
        assert z.shape[1] == self.z_dim
        z = z.view(-1, self.z_dim)
        en = self.energy(z).squeeze(1)
        return en


class _inference_svhn_4layer_v1(nn.Module):
    def __init__(self, z_dim, nif):
        super().__init__()
        self.nif = nif
        self.z_dim = z_dim
        self.ladder0 = nn.Sequential(
            conv2d_relu(3, nif[1], downsample=True),
            conv2d_relu(nif[1], nif[1], downsample=False),
            reshape(nif[1]*16*16),
        )
        self.ladder0_mean = nn.Linear(nif[1]*16*16, z_dim[0])
        self.ladder0_logvar = nn.Linear(nif[1]*16*16, z_dim[0])

        self.inference0 = nn.Sequential(
            conv2d_relu(3, nif[1], downsample=True),
            conv2d_relu(nif[1], nif[1], downsample=False)
        )

        self.ladder1 = nn.Sequential(
            conv2d_relu(nif[1], nif[2], downsample=True),
            conv2d_relu(nif[2], nif[2], downsample=False),
            reshape(nif[2]*8*8)
        )
        self.ladder1_mean = nn.Linear(nif[2]*8*8, z_dim[1])
        self.ladder1_logvar = nn.Linear(nif[2]*8*8, z_dim[1])

        self.inference1 = nn.Sequential(
            conv2d_relu(nif[1], nif[2], downsample=True),
            conv2d_relu(nif[2], nif[2], downsample=False),
        )

        self.ladder2 = nn.Sequential(
            conv2d_relu(nif[2], nif[3], downsample=True),
            conv2d_relu(nif[3], nif[3], downsample=False),
            reshape(nif[3]*4*4)
        )
        self.ladder2_mean = nn.Linear(nif[3]*4*4, z_dim[2])
        self.ladder2_logvar = nn.Linear(nif[3]*4*4, z_dim[2])

        self.inference2 = nn.Sequential(
            conv2d_relu(nif[2], nif[3], downsample=True),
            conv2d_relu(nif[3], nif[3], downsample=False),
        )

        self.ladder3 = nn.Sequential(
            reshape(nif[3]*4*4),
            fc_relu(nif[3]*4*4, nif[4]),
            fc_relu(nif[4], nif[4])
        )
        self.ladder3_mean = nn.Linear(nif[4], z_dim[3])
        self.ladder3_logvar = nn.Linear(nif[4], z_dim[3])

    def forward(self, x):
        z0_hidden = self.ladder0(x)
        z0_mean = self.ladder0_mean(z0_hidden)
        z0_logvar = self.ladder0_logvar(z0_hidden)

        ilatent1 = self.inference0(x)
        z1_hidden = self.ladder1(ilatent1)
        z1_mean = self.ladder1_mean(z1_hidden)
        z1_logvar = self.ladder1_logvar(z1_hidden)

        ilatent2 = self.inference1(ilatent1)
        z2_hidden = self.ladder2(ilatent2)
        z2_mean = self.ladder2_mean(z2_hidden)
        z2_logvar = self.ladder2_logvar(z2_hidden)

        ilatent3 = self.inference2(ilatent2)
        z3_hidden = self.ladder3(ilatent3)
        z3_mean = self.ladder3_mean(z3_hidden)
        z3_logvar = self.ladder3_logvar(z3_hidden)

        return z0_mean, z0_logvar, z1_mean, z1_logvar, z2_mean, z2_logvar, z3_mean, z3_logvar


class _generation_svhn_4layers_v1(nn.Module):
    def __init__(self, z_dim, ngf):
        super().__init__()
        self.ngf = ngf
        self.z_dim = z_dim

        self.gen3 = nn.Sequential(
            fc_relu(z_dim[3], ngf[4]),
            fc_relu(ngf[4], ngf[4]),
            fc_relu(ngf[4], ngf[4])
        )

        self.latent2_act = nn.Identity()

        self.gen2 = nn.Sequential(
            fc_relu(z_dim[2]+ngf[4], 4*4*ngf[3]),
            reshape(ngf[3], 4, 4),
            conv2d_t_relu(ngf[3], ngf[3], upsample=True),
            conv2d_t_relu(ngf[3], ngf[2], upsample=False),
        )# 8x8

        self.latent1_act = nn.Sequential(
            fc_relu(z_dim[1], 8*8*ngf[2]),
            reshape(ngf[2], 8, 8),
        )

        self.gen1 = nn.Sequential(
            conv2d_t_relu(ngf[2]+ngf[2], ngf[2], upsample=True),
            conv2d_t_relu(ngf[2], ngf[1], upsample=False),
        )#16x16

        self.latent0_act = nn.Sequential(
            fc_relu(z_dim[0], 16*16*ngf[1]),
            reshape(ngf[1], 16, 16),
        )

        self.gen0 = nn.Sequential(
            conv2d_t_relu(ngf[1]+ngf[1], ngf[1], upsample=True),
            conv2d_t_relu(ngf[1], 3, upsample=False),
            nn.Tanh()
        )

    def forward(self, zs):
        z0, z1, z2, z3 = zs[0], zs[1], zs[2], zs[3]
        tlatent3 = self.gen3(z3)
        z2_activated = self.latent2_act(z2)
        tlatent2 = self.gen2(t.cat([tlatent3, z2_activated], dim=1))

        z1_activated = self.latent1_act(z1)
        tlatent1 = self.gen1(t.cat([tlatent2, z1_activated], dim=1))

        z0_activated = self.latent0_act(z0)
        tlatent0 = self.gen0(t.cat([tlatent1, z0_activated], dim=1))

        out = tlatent0
        return out


class _inference_svhn_4layer_v2(nn.Module):
    def __init__(self, z_dim, nif):
        super().__init__()
        self.nif = nif
        self.z_dim = z_dim
        self.ladder0 = nn.Sequential(
            conv2d_relu(3, nif[1], downsample=True),
            conv2d_relu(nif[1], nif[1], downsample=False),
            reshape(nif[1]*16*16),
        )
        self.ladder0_mean = nn.Linear(nif[1]*16*16, z_dim[0])
        self.ladder0_logvar = nn.Linear(nif[1]*16*16, z_dim[0])

        self.inference0 = nn.Sequential(
            conv2d_relu(3, nif[1], downsample=True),
            conv2d_relu(nif[1], nif[1], downsample=False)
        )

        self.ladder1 = nn.Sequential(
            conv2d_relu(nif[1], nif[2], downsample=True),
            conv2d_relu(nif[2], nif[2], downsample=False),
            reshape(nif[2]*8*8)
        )
        self.ladder1_mean = nn.Linear(nif[2]*8*8, z_dim[1])
        self.ladder1_logvar = nn.Linear(nif[2]*8*8, z_dim[1])

        self.inference1 = nn.Sequential(
            conv2d_relu(nif[1], nif[2], downsample=True),
            conv2d_relu(nif[2], nif[2], downsample=False),
        )

        self.ladder2 = nn.Sequential(
            conv2d_relu(nif[2], nif[3], downsample=True),
            conv2d_relu(nif[3], nif[3], downsample=False),
            reshape(nif[3]*4*4)
        )
        self.ladder2_mean = nn.Linear(nif[3]*4*4, z_dim[2])
        self.ladder2_logvar = nn.Linear(nif[3]*4*4, z_dim[2])

        self.inference2 = nn.Sequential(
            conv2d_relu(nif[2], nif[3], downsample=True),
            conv2d_relu(nif[3], nif[3], downsample=False),
        )

        self.ladder3 = nn.Sequential(
            reshape(nif[3]*4*4),
            fc_relu(nif[3]*4*4, nif[4]),
            fc_relu(nif[4], nif[4])
        )
        self.ladder3_mean = nn.Linear(nif[4], z_dim[3])
        self.ladder3_logvar = nn.Linear(nif[4], z_dim[3])

    def forward(self, x):
        z0_hidden = self.ladder0(x)
        z0_mean = self.ladder0_mean(z0_hidden)
        z0_logvar = self.ladder0_logvar(z0_hidden)

        ilatent1 = self.inference0(x)
        z1_hidden = self.ladder1(ilatent1)
        z1_mean = self.ladder1_mean(z1_hidden)
        z1_logvar = self.ladder1_logvar(z1_hidden)

        ilatent2 = self.inference1(ilatent1)
        z2_hidden = self.ladder2(ilatent2)
        z2_mean = self.ladder2_mean(z2_hidden)
        z2_logvar = self.ladder2_logvar(z2_hidden)

        ilatent3 = self.inference2(ilatent2)
        z3_hidden = self.ladder3(ilatent3)
        z3_mean = self.ladder3_mean(z3_hidden)
        z3_logvar = self.ladder3_logvar(z3_hidden)

        return z0_mean, z0_logvar, z1_mean, z1_logvar, z2_mean, z2_logvar, z3_mean, z3_logvar


class _generation_svhn_4layers_v2(nn.Module):
    def __init__(self, z_dim, ngf):
        super().__init__()
        self.ngf = ngf
        self.z_dim = z_dim

        self.gen3 = nn.Sequential(
            nn.Linear(z_dim[3], ngf[4]),
            nn.LeakyReLU(0.2),
            nn.Linear(ngf[4], ngf[4]),
            nn.LeakyReLU(0.2),
            nn.Linear(ngf[4], ngf[4]),
        )

        self.latent2_act = nn.Sequential(
            nn.Linear(z_dim[2], ngf[4]),
            # nn.LeakyReLU(0.2),
            # nn.Linear(ngf[4], ngf[4]),
        )

        self.gen2 = nn.Sequential(
            nn.Linear(ngf[4], 4*4*ngf[3]),
            nn.LeakyReLU(0.2),
            reshape(ngf[3], 4, 4),
            nn.ConvTranspose2d(ngf[3], ngf[3], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[3], ngf[2], 3, 1, 1, bias=False),
        )# 8x8

        self.latent1_act = nn.Sequential(
            nn.Linear(z_dim[1], 8*8*ngf[2]),
            # nn.LeakyReLU(0.2),
            reshape(ngf[2], 8, 8),
        )

        self.gen1 = nn.Sequential(
            nn.ConvTranspose2d(ngf[2], ngf[2], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[2], ngf[1], 3, 1, 1, bias=False),
        )#16x16

        self.latent0_act = nn.Sequential(
            nn.Linear(z_dim[0], 16*16*ngf[1]),
            # nn.LeakyReLU(0.2),############
            reshape(ngf[1], 16, 16),
        )

        self.gen0 = nn.Sequential(
            nn.ConvTranspose2d(ngf[1], ngf[1], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[1], 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, zs):
        z0, z1, z2, z3 = zs[0], zs[1], zs[2], zs[3]
        tlatent3 = self.gen3(z3)
        z2_activated = self.latent2_act(z2)
        tlatent2 = self.gen2(tlatent3+z2_activated)

        z1_activated = self.latent1_act(z1)
        tlatent1 = self.gen1(tlatent2+z1_activated)

        z0_activated = self.latent0_act(z0)
        tlatent0 = self.gen0(tlatent1+z0_activated)

        out = tlatent0
        return out


class _generation_svhn_4layers_v3(nn.Module):
    def __init__(self, z_dim, ngf):
        super().__init__()
        self.ngf = ngf
        self.z_dim = z_dim

        self.gen3 = nn.Sequential(
            nn.Linear(z_dim[3], 2*2*ngf[4]),
            nn.LeakyReLU(0.2),
            nn.Linear(2*2*ngf[4], 2*2*ngf[4]),
            nn.LeakyReLU(0.2),
            reshape(ngf[4], 2, 2),
            nn.ConvTranspose2d(ngf[4], ngf[4], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[4], ngf[3], 3, 1, 1, bias=False),
        )

        self.latent2_act = nn.Sequential(
            nn.Linear(z_dim[2], 4*4*ngf[3]),
            reshape(ngf[3], 4, 4),
            # nn.LeakyReLU(0.2),
            # nn.Linear(ngf[4], ngf[4]),
        )

        self.gen2 = nn.Sequential(
            nn.ConvTranspose2d(ngf[3], ngf[3], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[3], ngf[2], 3, 1, 1, bias=False),
        )# 8x8

        self.latent1_act = nn.Sequential(
            nn.Linear(z_dim[1], 8*8*ngf[2]),
            # nn.LeakyReLU(0.2),
            reshape(ngf[2], 8, 8),
        )

        self.gen1 = nn.Sequential(
            nn.ConvTranspose2d(ngf[2], ngf[2], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[2], ngf[1], 3, 1, 1, bias=False),
        )#16x16

        self.latent0_act = nn.Sequential(
            nn.Linear(z_dim[0], 16*16*ngf[1]),
            # nn.LeakyReLU(0.2),############
            reshape(ngf[1], 16, 16),
        )

        self.gen0 = nn.Sequential(
            nn.ConvTranspose2d(ngf[1], ngf[1], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[1], 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, zs):
        z0, z1, z2, z3 = zs[0], zs[1], zs[2], zs[3]
        tlatent3 = self.gen3(z3)
        z2_activated = self.latent2_act(z2)
        tlatent2 = self.gen2(tlatent3+z2_activated)

        z1_activated = self.latent1_act(z1)
        tlatent1 = self.gen1(tlatent2+z1_activated)

        z0_activated = self.latent0_act(z0)
        tlatent0 = self.gen0(tlatent1+z0_activated)

        out = tlatent0
        return out


class _generation_svhn_4layers_v4(nn.Module):
    def __init__(self, z_dim, ngf):
        super().__init__()
        self.ngf = ngf
        self.z_dim = z_dim

        self.gen3 = nn.Sequential(
            nn.Linear(z_dim[3], ngf[4]),
            nn.LeakyReLU(0.2),
            nn.Linear(ngf[4], ngf[4]),
            nn.LeakyReLU(0.2),
            nn.Linear(ngf[4], ngf[3]),
        )

        self.latent2_act = nn.Sequential(
            nn.Linear(z_dim[2], ngf[3]),
            # nn.LeakyReLU(0.2),
            # nn.Linear(ngf[4], ngf[4]),
        )

        self.gen2 = nn.Sequential(
            nn.Linear(ngf[3], 4*4*ngf[3]),
            nn.LeakyReLU(0.2),
            reshape(ngf[3], 4, 4),
            nn.ConvTranspose2d(ngf[3], ngf[3], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[3], ngf[2], 3, 1, 1, bias=False),
        )# 8x8

        self.latent1_act = nn.Sequential(
            nn.Linear(z_dim[1], 8*8*ngf[2]),
            # nn.LeakyReLU(0.2),
            reshape(ngf[2], 8, 8),
        )

        self.gen1 = nn.Sequential(
            nn.ConvTranspose2d(ngf[2], ngf[2], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[2], ngf[1], 3, 1, 1, bias=False),
        )#16x16

        self.latent0_act = nn.Sequential(
            nn.Linear(z_dim[0], 16*16*ngf[1]),
            # nn.LeakyReLU(0.2),############
            reshape(ngf[1], 16, 16),
        )

        self.gen0 = nn.Sequential(
            nn.ConvTranspose2d(ngf[1], ngf[1], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[1], 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, zs):
        z0, z1, z2, z3 = zs[0], zs[1], zs[2], zs[3]
        tlatent3 = self.gen3(z3)
        z2_activated = self.latent2_act(z2)
        tlatent2 = self.gen2(tlatent3+z2_activated)

        z1_activated = self.latent1_act(z1)
        tlatent1 = self.gen1(tlatent2+z1_activated)

        z0_activated = self.latent0_act(z0)
        tlatent0 = self.gen0(tlatent1+z0_activated)

        out = tlatent0
        return out


class _inference_svhn_5layer_fid_v1(nn.Module):
    def __init__(self, z_dim, nif):
        super().__init__()
        self.nif = nif
        self.z_dim = z_dim
        self.ladder0 = nn.Sequential(
            conv2d_relu(3, nif[1], downsample=True),
            conv2d_relu(nif[1], nif[1], downsample=False),
            reshape(nif[1]*16*16),
        )
        self.ladder0_mean = nn.Linear(nif[1]*16*16, z_dim[0])
        self.ladder0_logvar = nn.Linear(nif[1]*16*16, z_dim[0])

        self.inference0 = nn.Sequential(
            conv2d_relu(3, nif[1], downsample=True),
            conv2d_relu(nif[1], nif[1], downsample=False)
        )

        self.ladder1 = nn.Sequential(
            conv2d_relu(nif[1], nif[2], downsample=True),
            conv2d_relu(nif[2], nif[2], downsample=False),
            reshape(nif[2]*8*8)
        )
        self.ladder1_mean = nn.Linear(nif[2]*8*8, z_dim[1])
        self.ladder1_logvar = nn.Linear(nif[2]*8*8, z_dim[1])

        self.inference1 = nn.Sequential(
            conv2d_relu(nif[1], nif[2], downsample=True),
            conv2d_relu(nif[2], nif[2], downsample=False),
        )

        self.ladder2 = nn.Sequential(
            conv2d_relu(nif[2], nif[3], downsample=True),
            conv2d_relu(nif[3], nif[3], downsample=False),
            reshape(nif[3]*4*4)
        )
        self.ladder2_mean = nn.Linear(nif[3]*4*4, z_dim[2])
        self.ladder2_logvar = nn.Linear(nif[3]*4*4, z_dim[2])

        self.inference2 = nn.Sequential(
            conv2d_relu(nif[2], nif[3], downsample=True),
            conv2d_relu(nif[3], nif[3], downsample=False),
        )

        self.ladder3 = nn.Sequential(
            conv2d_relu(nif[3], nif[4], downsample=True),
            conv2d_relu(nif[4], nif[4], downsample=False),
            reshape(nif[4]*2*2)
        )
        self.ladder3_mean = nn.Linear(nif[4]*2*2, z_dim[3])
        self.ladder3_logvar = nn.Linear(nif[4]*2*2, z_dim[3])

        self.inference3 = nn.Sequential(
            conv2d_relu(nif[3], nif[4], downsample=True),
            conv2d_relu(nif[4], nif[4], downsample=False),
        )

        self.ladder4 = nn.Sequential(
            reshape(nif[4]*2*2),
            fc_relu(nif[4]*2*2, nif[5]),
            fc_relu(nif[5], nif[5])
        )
        self.ladder4_mean = nn.Linear(nif[5], z_dim[4])
        self.ladder4_logvar = nn.Linear(nif[5], z_dim[4])

    def forward(self, x):
        z0_hidden = self.ladder0(x)
        z0_mean = self.ladder0_mean(z0_hidden)
        z0_logvar = self.ladder0_logvar(z0_hidden)

        ilatent1 = self.inference0(x)
        z1_hidden = self.ladder1(ilatent1)
        z1_mean = self.ladder1_mean(z1_hidden)
        z1_logvar = self.ladder1_logvar(z1_hidden)

        ilatent2 = self.inference1(ilatent1)
        z2_hidden = self.ladder2(ilatent2)
        z2_mean = self.ladder2_mean(z2_hidden)
        z2_logvar = self.ladder2_logvar(z2_hidden)

        ilatent3 = self.inference2(ilatent2)
        z3_hidden = self.ladder3(ilatent3)
        z3_mean = self.ladder3_mean(z3_hidden)
        z3_logvar = self.ladder3_logvar(z3_hidden)

        ilatent4 = self.inference3(ilatent3)
        z4_hidden = self.ladder4(ilatent4)
        z4_mean = self.ladder4_mean(z4_hidden)
        z4_logvar = self.ladder4_logvar(z4_hidden)

        return z0_mean, z0_logvar, z1_mean, z1_logvar, z2_mean, z2_logvar, z3_mean, z3_logvar, z4_mean, z4_logvar

class _generation_svhn_5layers_fid_v1(nn.Module):
    def __init__(self, z_dim, ngf):
        super().__init__()
        self.ngf = ngf
        self.z_dim = z_dim
        # self.fs = [self.data_dims[0], self.data_dims[0] // 2, self.data_dims[0] // 4, self.data_dims[0] // 8,
        #            self.data_dims[0] // 16]
        self.gen4 = nn.Sequential(
            nn.Linear(z_dim[4], ngf[6]),
            nn.LeakyReLU(0.2),
            nn.Linear(ngf[6], ngf[6]),
            nn.LeakyReLU(0.2),
            nn.Linear(ngf[6], ngf[5]*2*2),
            nn.LeakyReLU(0.2),
            reshape(ngf[5], 2, 2)
        )

        self.latent3_act = nn.Sequential(
            nn.Linear(z_dim[3], 2*2*ngf[5]),
            nn.LeakyReLU(0.2),
            reshape(ngf[5], 2, 2)
            # nn.LeakyReLU(0.2),
            # nn.Linear(ngf[4], ngf[4]),
        )

        self.gen3 = nn.Sequential(
            nn.ConvTranspose2d(ngf[5]*2, ngf[5], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[5], ngf[4], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[4], ngf[4], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2),
        )

        self.latent2_act = nn.Sequential(
            nn.Linear(z_dim[2], 4*4*ngf[4]),
            nn.LeakyReLU(0.2),
            reshape(ngf[4], 4, 4),
            # nn.LeakyReLU(0.2),
            # nn.Linear(ngf[4], ngf[4]),
        )

        self.gen2 = nn.Sequential(
            nn.ConvTranspose2d(ngf[4]*2, ngf[4], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[4], ngf[3], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[3], ngf[3], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2),
        )# 8x8

        self.latent1_act = nn.Sequential(
            nn.Linear(z_dim[1], 8*8*ngf[3]),
            nn.LeakyReLU(0.2),
            reshape(ngf[3], 8, 8),
        )

        self.gen1 = nn.Sequential(
            nn.ConvTranspose2d(ngf[3]*2, ngf[3], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[3], ngf[2], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[2], ngf[2], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2),
        )#16x16

        self.latent0_act = nn.Sequential(
            nn.Linear(z_dim[0], 16*16*ngf[2]),
            nn.LeakyReLU(0.2),############
            reshape(ngf[2], 16, 16),
        )

        self.gen0 = nn.Sequential(
            nn.ConvTranspose2d(ngf[2]*2, ngf[2], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[2], ngf[1], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[1], ngf[1], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf[1], 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, zs):
        z0, z1, z2, z3, z4 = zs[0], zs[1], zs[2], zs[3], zs[4]
        tlatent4 = self.gen4(z4)
        z3_activated = self.latent3_act(z3)
        tlatent3 = self.gen3(t.cat([tlatent4, z3_activated], dim=1))
        # tlatent3 = self.gen3(tlatent4+z3_activated)

        z2_activated = self.latent2_act(z2)
        tlatent2 = self.gen2(t.cat([tlatent3, z2_activated], dim=1))
        # tlatent2 = self.gen2(tlatent3+z2_activated)

        z1_activated = self.latent1_act(z1)
        tlatent1 = self.gen1(t.cat([tlatent2, z1_activated], dim=1))
        # tlatent1 = self.gen1(tlatent2+z1_activated)

        z0_activated = self.latent0_act(z0)
        tlatent0 = self.gen0(t.cat([tlatent1, z0_activated], dim=1))
        # tlatent0 = self.gen0(tlatent1+z0_activated)

        out = tlatent0
        return out

