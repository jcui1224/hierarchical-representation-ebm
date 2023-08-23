from nets import *
from utils import *
import os.path
import matplotlib.pyplot as plt
parameters = {
                    # 'axes.labelsize': 15,
                    # 'axes.titlesize': 25,
                    # 'figure.titlesize': 15,
                    # 'legend.fontsize': 15,
                    'xtick.labelsize': 10,
                    'ytick.labelsize': 10,
                    "savefig.bbox": 'tight',
}
plt.rcParams.update(parameters)
from matplotlib import image
import torchvision
import torchvision.transforms as transforms
import torch as t
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import random
import itertools
import datetime
import argparse
from collections import OrderedDict

exp_path = f'./'

def show_multiple_images(img_list, path_dir, imgs_dir, labels=None):
    def show(imgs, path, labels):
        if not isinstance(imgs, list):
            imgs = [imgs]
        if not isinstance(labels, list):
            labels = [[]] * len(img_list)
        fig, axs = plt.subplots(ncols=len(imgs), figsize=(40, 40*len(img_list)), squeeze=False)
        for i, img in enumerate(imgs):
            img = image.imread(img)
            axs[0, i].imshow(img)
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[0, i].set_title(labels[i])

        plt.savefig(path)
        plt.close()

    nrow = {8:4, 16: 4, 32:8, 64:8, 100:10}[img_list[0].shape[0]]
    path_list = []
    for i, img in enumerate(img_list):
        vutils.save_image(img, path_dir + f'temp_{i}.png', nrow=nrow, normalize=True)
        path_list.append(path_dir + f'temp_{i}.png')
    show(path_list, path=imgs_dir, labels=labels)


def sample_z0(args):
    z0 = t.randn(args.batch_size, args.z0_dim).to(args.device)
    z1 = t.randn(args.batch_size, args.z1_dim).to(args.device)
    z2 = t.randn(args.batch_size, args.z2_dim).to(args.device)
    z3 = t.randn(args.batch_size, args.z3_dim).to(args.device)
    return [z0, z1, z2, z3]


def reparametrize(mu, log_sigma, is_train=True):
    if is_train:
        std = t.exp(log_sigma.mul(0.5))
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu


def langevin_prior(zs_init, netE, args, should_print=True):
    e_l_steps = args.e_l_steps
    e_l_step_size = args.e_l_step_size
    zs = []
    for z in zs_init:
        z = z.clone().detach()
        z.requires_grad = True
        zs.append(z)

    for k in range(e_l_steps):
        en = netE(zs).sum()
        en_grad = t.autograd.grad(en, zs)

        for i, z in enumerate(zs):
            z.data += 0.5 * e_l_step_size * e_l_step_size * (en_grad[i] - z.data)

            z.data += e_l_step_size * t.randn_like(z)

        if (k % 5 == 0 or k == e_l_steps - 1) and should_print:
            logging.info('Langevin Prior {}/{}: EN: {}'.format(k + 1, e_l_steps, en.item()))

    zs_return = []
    for z in zs:
        z = z.detach()
        zs_return.append(z)

    return zs_return


def hierarchical_sampling(netG, netE, args):
    zs_init = sample_z0(args)
    zs_ebm = langevin_prior(zs_init, netE, args, should_print=False)
    fixed_zs = []
    for z in zs_ebm:
        fixed_z = z[0].repeat(z.shape[0], 1)
        fixed_zs.append(fixed_z)

    samples = []
    for i, z in enumerate(zs_ebm):
        vis_zs = fixed_zs.copy()
        vis_zs[i] = z
        samples.append(netG(vis_zs).detach())
    return samples


def sample_x(netG, netE, args):
    zs_init = sample_z0(args)
    zs_ebm = langevin_prior(zs_init, netE, args, should_print=False)
    samples = netG(zs_ebm)
    return samples


def Start(netG, netE, netI, args, logger):

    batch_num = 10

    for i in range(batch_num):
        hs_imgs_dir = args.dir + 'imgs/hierarchical_sampling/'
        os.makedirs(hs_imgs_dir, exist_ok=True)
        samples = hierarchical_sampling(netG, netE, args)
        show_multiple_images(samples, path_dir=hs_imgs_dir, imgs_dir=hs_imgs_dir + f'{i}.png', labels=None)
    return

def build_netG(args):
    class _generation_4layers(nn.Module):
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
            )

            self.gen2 = nn.Sequential(
                nn.Linear(ngf[3], 4 * 4 * ngf[3]),
                nn.LeakyReLU(0.2),
                reshape(ngf[3], 4, 4),
                nn.ConvTranspose2d(ngf[3], ngf[3], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(ngf[3], ngf[2], 3, 1, 1, bias=False),
            )

            self.latent1_act = nn.Sequential(
                nn.Linear(z_dim[1], 8 * 8 * ngf[2]),
                reshape(ngf[2], 8, 8),
            )

            self.gen1 = nn.Sequential(
                nn.ConvTranspose2d(ngf[2], ngf[2], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(ngf[2], ngf[1], 3, 1, 1, bias=False),
            )

            self.latent0_act = nn.Sequential(
                nn.Linear(z_dim[0], 16 * 16 * ngf[1]),
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
            tlatent2 = self.gen2(tlatent3 + z2_activated)

            z1_activated = self.latent1_act(z1)
            tlatent1 = self.gen1(tlatent2 + z1_activated)

            z0_activated = self.latent0_act(z0)
            tlatent0 = self.gen0(tlatent1 + z0_activated)

            out = tlatent0
            return out

    netG = _generation_4layers(z_dim=[args.z0_dim, args.z1_dim, args.z2_dim, args.z3_dim], ngf=args.cs)
    netG.apply(weights_init)
    netG.to(args.device)
    return netG


def build_netE(args):
    class energy_model(nn.Module):
        def __init__(self, z_dim, num_layers, ndf):
            super().__init__()
            self.z_dim = z_dim
            current_dims = self.z_dim
            layers = OrderedDict()
            for i in range(num_layers):
                layers['fc{}'.format(i + 1)] = nn.Linear(current_dims, ndf)
                layers['lrelu{}'.format(i + 1)] = nn.LeakyReLU(0.2)
                current_dims = ndf

            layers['out'] = nn.Linear(current_dims, 1)
            self.energy = nn.Sequential(layers)

        def forward(self, zs):
            z = t.cat(zs, dim=1)
            assert z.shape[1] == self.z_dim
            z = z.view(-1, self.z_dim)
            en = self.energy(z).squeeze(1)
            return en

    netE = energy_model(z_dim=sum([args.z0_dim, args.z1_dim, args.z2_dim, args.z3_dim]), num_layers=args.en_layers, ndf=args.en_ndf)
    netE.apply(weights_init)
    netE.to(args.device)
    netE = add_sn(netE)
    return netE


def build_netI(args):
    class _inference_4layer(nn.Module):
        def __init__(self, z_dim, nif):
            super().__init__()
            self.nif = nif
            self.z_dim = z_dim
            self.ladder0 = nn.Sequential(
                conv2d_relu(3, nif[1], downsample=True),
                conv2d_relu(nif[1], nif[1], downsample=False),
                reshape(nif[1] * 16 * 16),
            )
            self.ladder0_mean = nn.Linear(nif[1] * 16 * 16, z_dim[0])
            self.ladder0_logvar = nn.Linear(nif[1] * 16 * 16, z_dim[0])

            self.inference0 = nn.Sequential(
                conv2d_relu(3, nif[1], downsample=True),
                conv2d_relu(nif[1], nif[1], downsample=False)
            )

            self.ladder1 = nn.Sequential(
                conv2d_relu(nif[1], nif[2], downsample=True),
                conv2d_relu(nif[2], nif[2], downsample=False),
                reshape(nif[2] * 8 * 8)
            )
            self.ladder1_mean = nn.Linear(nif[2] * 8 * 8, z_dim[1])
            self.ladder1_logvar = nn.Linear(nif[2] * 8 * 8, z_dim[1])

            self.inference1 = nn.Sequential(
                conv2d_relu(nif[1], nif[2], downsample=True),
                conv2d_relu(nif[2], nif[2], downsample=False),
            )

            self.ladder2 = nn.Sequential(
                conv2d_relu(nif[2], nif[3], downsample=True),
                conv2d_relu(nif[3], nif[3], downsample=False),
                reshape(nif[3] * 4 * 4)
            )
            self.ladder2_mean = nn.Linear(nif[3] * 4 * 4, z_dim[2])
            self.ladder2_logvar = nn.Linear(nif[3] * 4 * 4, z_dim[2])

            self.inference2 = nn.Sequential(
                conv2d_relu(nif[2], nif[3], downsample=True),
                conv2d_relu(nif[3], nif[3], downsample=False),
            )

            self.ladder3 = nn.Sequential(
                reshape(nif[3] * 4 * 4),
                fc_relu(nif[3] * 4 * 4, nif[4]),
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

    netI = _inference_4layer(z_dim=[args.z0_dim, args.z1_dim, args.z2_dim, args.z3_dim], nif=args.cs)
    netI.apply(weights_init)
    netI.to(args.device)

    return netI

def letgo(output_dir):

    set_seeds(1234)
    loaded_config = load_args(exp_path)
    loaded_config = argparse.Namespace(**loaded_config)

    args = parse_args()
    args = overwrite_opt(loaded_config, vars(args))

    output_dir += '/'
    args.dir = output_dir

    [os.makedirs(args.dir + f'{f}/', exist_ok=True) for f in ['imgs']]

    logger = Logger(args.dir, f"job0")
    logger.info('Config')
    logger.info(args)

    save_args(vars(args), output_dir)

    netG = build_netG(args)
    netE = build_netE(args)
    netI = build_netI(args)

    ckpt = t.load(exp_path + 'ckpt.pth', map_location='cpu')
    netG.load_state_dict(ckpt['netG'], strict=True)
    netE.load_state_dict(ckpt['netE'], strict=True)
    netI.load_state_dict(ckpt['netI'], strict=True)

    logger.info(f"netG params: {compute_model_params(netG)}")
    logger.info(f"netE params: {compute_model_params(netE)}")
    logger.info(f"netI params: {compute_model_params(netI)}")

    Start(netG, netE, netI, args, logger)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--e_l_steps', type=int, default=60) # update it
    parser.add_argument('--e_l_step_size', type=float, default=4e-1)

    parser.add_argument('--device', type=int, default=0)
    # Parser
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print("Invalid arguments %s" % unknown)
        parser.print_help()
        sys.exit()
    return args

def set_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def main():
    output_dir = './{}/'.format(os.path.splitext(os.path.basename(__file__))[0])
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    output_dir += t + '/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + 'code/', exist_ok=True)

    [save_file(output_dir, f) for f in ['nets.py', 'utils.py', os.path.basename(__file__)]]

    letgo(output_dir)

if __name__ == '__main__':
    main()
