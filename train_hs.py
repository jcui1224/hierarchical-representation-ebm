from nets import *
from utils import *
import torchvision
import torchvision.transforms as transforms
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

mse = nn.MSELoss(reduction='sum').cuda()
svhn_config = {
    'dataset': 'svhn',
    'img_size': 32,
    'normalize_data': True,
}


def get_dataset(args):
    data_dir = '/data4/jcui7/images/data/' if 'Tian-ds' not in __file__ else '/Tian-ds/jcui7/HugeData/'
    transform = transforms.Compose(
        [transforms.Resize(args.img_size), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds_train = torchvision.datasets.SVHN(data_dir + 'svhn', download=True, split='extra', transform=transform)
    ds_val = torchvision.datasets.SVHN(data_dir + 'svhn', download=True, split='test', transform=transform)
    input_shape = [3, args.img_size, args.img_size]
    return ds_train, ds_val, input_shape


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


def diag_normal_NLL(z, z_mu, z_log_sigma):
    # define the Negative Log Probability of Normal which has diagonal cov
    # input: [batch nz, 1, 1] squeeze it to batch nz
    # return: shape is [batch]
    nll = 0.5 * t.sum(z_log_sigma.squeeze(), dim=1) + \
          0.5 * t.sum((t.mul(z - z_mu, z - z_mu) / (1e-6 + t.exp(z_log_sigma))).squeeze(), dim=1)
    return nll.squeeze()


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


def recon_x(x, netG, netI):
    z0_mean, z0_logvar, z1_mean, z1_logvar, z2_mean, z2_logvar, z3_mean, z3_logvar = netI(x)
    z0 = reparametrize(z0_mean, z0_logvar)
    z1 = reparametrize(z1_mean, z1_logvar)
    z2 = reparametrize(z2_mean, z2_logvar)
    z3 = reparametrize(z3_mean, z3_logvar)
    recon = netG([z0, z1, z2, z3]).detach()
    return recon


def fit(netG, netE, netI, dl_train, test_batch, args, logger):

    optE = t.optim.Adam(netE.parameters(), lr=args.lrE, weight_decay=args.e_decay, betas=(args.beta1E, 0.9))
    opt = t.optim.Adam(list(netI.parameters())+list(netG.parameters()), lr=args.lrG, weight_decay=args.g_decay, betas=(args.beta1G, 0.9))
    lrE_schedule = t.optim.lr_scheduler.ExponentialLR(optE, args.e_gamma)
    lr_schedule = t.optim.lr_scheduler.ExponentialLR(opt, args.i_gamma)
    log_iter = 500

    for ep in range(args.epochs):
        lrE_schedule.step(epoch=ep)
        lr_schedule.step(epoch=ep)

        for i, x in enumerate(dl_train, 0):
            training_log = f"[{ep}/{args.epochs}][{i}/{len(dl_train)}] \n"

            x = x[0].to(args.device) if type(x) is list else x.to(args.device)
            batch_size = x.shape[0]

            opt.zero_grad()
            z0_q_mean, z0_q_logvar, z1_q_mean, z1_q_logvar, z2_q_mean, z2_q_logvar, z3_q_mean, z3_q_logvar = netI(x)
            z0_q = reparametrize(z0_q_mean, z0_q_logvar)
            z1_q = reparametrize(z1_q_mean, z1_q_logvar)
            z2_q = reparametrize(z2_q_mean, z2_q_logvar)
            z3_q = reparametrize(z3_q_mean, z3_q_logvar)

            z0_kl = t.mean(diag_normal_NLL(z0_q, t.zeros_like(z0_q), t.zeros_like(z0_q)) - diag_normal_NLL(z0_q, z0_q_mean, z0_q_logvar))
            z1_kl = t.mean(diag_normal_NLL(z1_q, t.zeros_like(z1_q), t.zeros_like(z1_q)) - diag_normal_NLL(z1_q, z1_q_mean, z1_q_logvar))
            z2_kl = t.mean(diag_normal_NLL(z2_q, t.zeros_like(z2_q), t.zeros_like(z2_q)) - diag_normal_NLL(z2_q, z2_q_mean, z2_q_logvar))
            z3_kl = t.mean(diag_normal_NLL(z3_q, t.zeros_like(z3_q), t.zeros_like(z3_q)) - diag_normal_NLL(z3_q, z3_q_mean, z3_q_logvar))
            kl = z0_kl + z1_kl + z2_kl + z3_kl

            et = t.mean(netE([z0_q, z1_q, z2_q, z3_q]))
            rec = netG([z0_q, z1_q, z2_q, z3_q])
            rec_loss = mse(rec, x) / batch_size

            loss = args.Gfactor * rec_loss + kl - et
            training_log += f'rec: {rec_loss.item():.3f} z0_KL: {z0_kl.item():.3f} z1_KL: {z1_kl.item():.3f} ' \
                            f'z2_KL: {z2_kl.item():.3f} z3_KL: {z2_kl.item():.3f}\n'
            loss.backward()
            opt.step()

            optE.zero_grad()
            z_e_0 = sample_z0(args)
            z_e_k = langevin_prior(z_e_0, netE, args, should_print=(i % log_iter == 0))
            et = t.mean(netE([z0_q.detach(), z1_q.detach(), z2_q.detach(), z3_q.detach()]))
            ef = t.mean(netE(z_e_k))
            e_loss = ef - et
            e_loss.backward()
            optE.step()
            training_log += f'et: {et.item():.3f} ef: {ef.item():.3f}'

            if i % log_iter == 0:
                logger.info(training_log)
                hs_imgs_dir = args.dir + 'imgs/hierarchical_sampling/'
                os.makedirs(hs_imgs_dir, exist_ok=True)
                samples = hierarchical_sampling(netG, netE, args)
                show_multiple_images(samples, path_dir=hs_imgs_dir, imgs_dir=hs_imgs_dir + f'{ep * len(dl_train) + i:>07d}.png', labels=None)
                rec_imgs_dir = args.dir + 'imgs/rec/'
                os.makedirs(rec_imgs_dir, exist_ok=True)
                show_single_batch(recon_x(test_batch, netG, netI), rec_imgs_dir + f'{ep * len(dl_train) + i:>07d}.png', nrow=10)
                syn_imgs_dir = args.dir + 'imgs/syn/'
                os.makedirs(syn_imgs_dir, exist_ok=True)
                show_single_batch(sample_x(netG, netE, args), syn_imgs_dir + f'{ep * len(dl_train) + i:>07d}.png', nrow=10)

                os.makedirs(args.dir + '/ckpt', exist_ok=True)
                save_dict = {
                    'epoch': ep,
                    'netG': netG.state_dict(),
                    'netE': netE.state_dict(),
                    'netI': netI.state_dict(),
                    'opt': opt.state_dict(),
                    'optE': optE.state_dict(),
                }
                t.save(save_dict, '{}/{}.pth'.format(args.dir + '/ckpt', ep * len(dl_train) + i))
                keep_last_ckpt(path=args.dir + '/ckpt/', num=30)
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


def letgo(args_job, output_dir):
    set_seeds(1234)
    args = parse_args()
    args = overwrite_opt(args, args_job)
    args = overwrite_opt(args, svhn_config)
    output_dir += '/'
    args.dir = output_dir

    [os.makedirs(args.dir + f'{f}/', exist_ok=True) for f in ['ckpt', 'imgs']]

    logger = Logger(args.dir, f"job0")
    logger.info('Config')
    logger.info(args)

    save_args(vars(args), output_dir)

    ds_train, ds_val, input_shape = get_dataset(args)
    dl_train = t.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    dl_val = t.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    args.input_shape = input_shape
    logger.info("Training samples %d" % len(ds_train))

    fix_x = next(iter(dl_train))
    test_batch = fix_x[0].to(args.device) if type(fix_x) is list else fix_x.to(args.device)
    show_single_batch(test_batch, args.dir + 'imgs/test_batch.png', nrow=int(args.batch_size ** 0.5))

    netG = build_netG(args)
    netE = build_netE(args)
    netI = build_netI(args)

    logger.info(f"netG params: {compute_model_params(netG)}")
    logger.info(f"netE params: {compute_model_params(netE)}")
    logger.info(f"netI params: {compute_model_params(netI)}")

    fit(netG, netE, netI, dl_train, test_batch, args, logger)

    return


def parse_args():
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--lrE', type=float, default=2e-6, help='learning rate for E, default=0.0002')
    parser.add_argument('--lrG', type=float, default=3e-4, help='learning rate for GI, default=0.0002')
    parser.add_argument('--lrI', type=float, default=1e-4, help='learning rate for GI, default=0.0002')

    parser.add_argument('--Gfactor', type=float, default=3.0)

    parser.add_argument('--en_layers', type=int, default=4)
    parser.add_argument('--en_ndf', type=int, default=50)
    parser.add_argument('--e_l_steps', type=int, default=60)
    parser.add_argument('--e_l_step_size', type=float, default=4e-1)

    parser.add_argument('--cs', type=list,  default=[3, 32, 64, 64, 256])

    parser.add_argument('--z0_dim', type=int, default=2, help='size of the latent z vector') # 100
    parser.add_argument('--z1_dim', type=int, default=2, help='size of the latent z vector') # 100
    parser.add_argument('--z2_dim', type=int, default=2, help='size of the latent z vector') # 100
    parser.add_argument('--z3_dim', type=int, default=10, help='size of the latent z vector') # 100

    parser.add_argument('--beta1E',  type=float, default=0., help='beta1 for adam. default=0.5')
    parser.add_argument('--beta1G',  type=float, default=0., help='beta1 for adam GI. default=0.5')
    parser.add_argument('--beta1I',  type=float, default=0., help='beta1 for adam GI. default=0.5')
    parser.add_argument('--e_decay', type=float, default=0.0000, help='weight decay for E')
    parser.add_argument('--i_decay', type=float, default=0.0005, help='weight decay for I')
    parser.add_argument('--g_decay', type=float, default=0.0005, help='weight decay for G')
    parser.add_argument('--e_gamma', type=float, default=0.998, help='lr decay for EBM')
    parser.add_argument('--i_gamma', type=float, default=0.998, help='lr decay for I')
    parser.add_argument('--g_gamma', type=float, default=0.998, help='lr decay for G')

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

    opt = dict()
    letgo(opt, output_dir)

if __name__ == '__main__':
    main()
