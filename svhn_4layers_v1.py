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
from config import svhn_config
from nets import weights_init
from dataset import get_dataset
from utils import *
from pygrid_utils import *
from torch.autograd import Variable
import numpy as np
import random
import itertools
import datetime
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
grid_log_init = {'Broken': 'False', 'Broken_epoch': 0, 'fid_best_syn': 0., 'fid_best_ep': 0}
mse = nn.MSELoss(reduction='sum').cuda()

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

    def eval_flag():
        netG.eval()
        netE.eval()
        netI.eval()
        # requires_grad(netG, False)
        # requires_grad(netE, False)
        # requires_grad(netI, False)

    def train_flag():
        netG.train()
        netE.train()
        netI.train()
        # requires_grad(netG, True)
        # requires_grad(netE, True)
        # requires_grad(netI, True)

    # optG = t.optim.Adam(netG.parameters(), lr=args.lrG, weight_decay=args.g_decay, betas=(args.beta1G, 0.9))
    optE = t.optim.Adam(netE.parameters(), lr=args.lrE, weight_decay=args.e_decay, betas=(args.beta1E, 0.9))
    opt = t.optim.Adam(list(netI.parameters())+list(netG.parameters()), lr=args.lrG, weight_decay=args.g_decay, betas=(args.beta1G, 0.9))
    lrE_schedule = t.optim.lr_scheduler.ExponentialLR(optE, args.e_gamma)
    lr_schedule = t.optim.lr_scheduler.ExponentialLR(opt, args.i_gamma)

    import math
    fid_best_syn = math.inf
    fid_best_ep = 0
    to_range_0_1 = lambda x: (x + 1.) / 2. if args.normalize_data else x
    log_iter = int(len(dl_train) // 2) if len(dl_train) > 1000 else int(len(dl_train) // 1)

    fid_ds = []
    for j in range(50000 // args.batch_size):
        fix_x = next(iter(dl_train))
        fid_batch = fix_x[0].to(args.device) if type(fix_x) is list else fix_x.to(args.device)
        fid_batch = to_range_0_1(fid_batch).clamp(min=0., max=1.)
        fid_ds.append(fid_batch)
    fid_ds = t.cat(fid_ds)

    for ep in range(args.epochs):
        lrE_schedule.step(epoch=ep)
        # lrG_schedule.step(epoch=ep)
        lr_schedule.step(epoch=ep)

        for i, x in enumerate(dl_train, 0):
            if i % log_iter == 0:
                logger.info(
                    "==" * 10 + f"ep: {ep} batch: [{i}/{len(dl_train)}] best_fid_syn: {fid_best_syn:.3f} best_fid_ep: {fid_best_ep}" + "==" * 10)

            training_log = f"[{ep}/{args.epochs}][{i}/{len(dl_train)}] best_fid_syn: {fid_best_syn:.3f} fid best ep: {fid_best_ep} \n"

            train_flag()
            x = x[0].to(args.device) if type(x) is list else x.to(args.device)
            batch_size = x.shape[0]

            opt.zero_grad()
            # optG.zero_grad()
            # optE.zero_grad()
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
            # optG.step()
            # optE.step()

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

                if not args.fid:
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

            if t.isnan(et.mean()) or t.isnan(ef.mean()) or e_loss.item() > 1e3 or e_loss.item() < -1e3:
                logger.info("Got NaN at ep {} iter {}".format(ep, i))
                logger.info(training_log)
                return ['True', ep, fid_best_syn, fid_best_ep]

        if args.fid and ep >= args.n_metrics_start and ep % args.n_metrics == 0:
            from pytorch_fid_jcui7.fid_score import compute_fid
            from tqdm import tqdm
            try:
                s1 = []
                for _ in tqdm(range(int(50000 / args.batch_size))):
                    syn = sample_x(netG, netE, args).detach()
                    syn_corr = to_range_0_1(syn).clamp(min=0., max=1.)
                    s1.append(syn_corr)
                s1 = t.cat(s1)
                fid = compute_fid(x_train=fid_ds, x_samples=s1, path=None)
                logger.info(f'fid gen: {fid:.5f}')

                if fid < fid_best_syn:
                    fid_best_syn = fid
                    fid_best_ep = ep
                    os.makedirs(args.dir + '/ckpt', exist_ok=True)
                    save_dict = {
                        'epoch': ep,
                        'netG': netG.state_dict(),
                        'netE': netE.state_dict(),
                        'netI': netI.state_dict(),
                        'opt': opt.state_dict(),
                        'optE': optE.state_dict(),
                    }
                    t.save(save_dict, '{}/{:.4f}_{}.pth'.format(args.dir + '/ckpt', fid, ep))

                # if fid > fid_best_syn + 5:
                #     return ['True', ep, fid_best_syn, fid_best_ep]

            except Exception as e:
                print(e)
                logger.critical(e, exc_info=True)

    return ['False', args.epochs, fid_best_syn, fid_best_ep]


def build_netG(args):
    if args.G_type == 'svhn_4layers_v1':
        from nets import _generation_svhn_4layers_v1 as _netG
        netG = _netG(z_dim=[args.z0_dim, args.z1_dim, args.z2_dim, args.z3_dim], ngf=args.cs)
        netG.apply(weights_init)
        netG.to(args.device)
    elif args.G_type == 'svhn_4layers_v2':
        from nets import _generation_svhn_4layers_v2 as _netG
        netG = _netG(z_dim=[args.z0_dim, args.z1_dim, args.z2_dim, args.z3_dim], ngf=args.cs)
        netG.apply(weights_init)
        netG.to(args.device)
    elif args.G_type == 'svhn_4layers_v3':
        from nets import _generation_svhn_4layers_v3 as _netG
        netG = _netG(z_dim=[args.z0_dim, args.z1_dim, args.z2_dim, args.z3_dim], ngf=args.cs)
        netG.apply(weights_init)
        netG.to(args.device)
    elif args.G_type == 'svhn_4layers_v4':
        from nets import _generation_svhn_4layers_v4 as _netG
        netG = _netG(z_dim=[args.z0_dim, args.z1_dim, args.z2_dim, args.z3_dim], ngf=args.cs)
        netG.apply(weights_init)
        netG.to(args.device)
    else:
        raise Exception("choose G type")
    return netG


def build_netE(args):
    if args.E_type == 'v1':
        from nets import energy_model as _netE
        netE = _netE(z_dim=sum([args.z0_dim, args.z1_dim, args.z2_dim, args.z3_dim]), num_layers=args.en_layers, ndf=args.en_ndf)
        netE.apply(weights_init)
        netE.to(args.device)
    else:
        raise Exception("choose E type")
    if args.add_sn:
        netE = add_sn(netE)
    return netE


def build_netI(args):
    if args.I_type == 'svhn_4layers_v1':
        from nets import _inference_svhn_4layer_v1 as _netI
        netI = _netI(z_dim=[args.z0_dim, args.z1_dim, args.z2_dim, args.z3_dim], nif=args.cs)
        netI.apply(weights_init)
        netI.to(args.device)
    else:
        raise Exception("choose I type")
    return netI


def letgo(args_job, output_dir, return_dict):

    set_seeds(1234)
    args = parse_args()
    args = overwrite_opt(args, args_job)
    args = overwrite_opt(args, svhn_config)
    output_dir += '/'
    args.dir = output_dir

    [os.makedirs(args.dir + f'{f}/', exist_ok=True) for f in ['ckpt', 'imgs']]

    logger = Logger(args.dir, f"job{args.job_id}")
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

    return_list = fit(netG, netE, netI, dl_train, test_batch, args, logger)

    log_stat = {}
    for grid_log_key, return_value in zip(grid_log_init.keys(), return_list):
        log_stat[grid_log_key] = return_value
    return_dict['stats'] = log_stat
    return

def update_job_result(job_opt, job_stats):
    # TODO add your result metric here
    for grid in grid_log_init.keys():
        job_opt[grid] = job_stats[grid]

def create_args_grid():
    # TODO add your enumeration of parameters here
    args_dict = {
        'add_sn': [True],
        'Gfactor': [3.0],
        'en_layers': [3, 4],
        'en_ndf': [50],
        'lrE': [2e-6],
        'e_l_steps': [60],
        'e_l_step_size': [0.4],
        'cs': [[3, 32, 64, 64, 256]],
        'z3_dim': [10],
        'z2_dim': [2],
        'z1_dim': [2],
        'z0_dim': [2],
    }
    args_list = list(args_dict.values())

    opt_list = []

    for i, args in enumerate(itertools.product(*args_list)):
        opt_job = {'job_id': int(i), 'status': 'open'}
        v = [args[i] for i in range(len(args_list))]
        k = args_dict.keys()
        opt_args = dict(zip(k, v))
        # opt_args.update({'mcmc_temp': (opt_args['e_n_step_size']/opt_args['e_l_step_size'])**2})
        # opt_args.update({'z0_dim': opt_args['z2_dim']})
        # opt_args.update({'z1_dim': opt_args['z2_dim']})
        # TODO add your result metric here
        opt_result = grid_log_init
        opt_list += [merge_dicts(opt_job, opt_args, opt_result)]

    return opt_list

def parse_args():
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--G_type', type=str, default='svhn_4layers_v4', help='svhn_4layers_v1, svhn_4layers_v2')
    parser.add_argument('--E_type', type=str, default='v1', help='v1')
    parser.add_argument('--I_type', type=str, default='svhn_4layers_v1', help='svhn_4layers_v1')

    parser.add_argument('--lrE', type=float, default=1e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--lrG', type=float, default=3e-4, help='learning rate for GI, default=0.0002')
    parser.add_argument('--lrI', type=float, default=1e-4, help='learning rate for GI, default=0.0002')

    parser.add_argument('--Gfactor', type=float, default=5.0)

    parser.add_argument('--en_layers', type=int, default=2)
    parser.add_argument('--en_ndf', type=int, default=100)
    parser.add_argument('--e_l_steps', type=int, default=60)
    parser.add_argument('--e_l_step_size', type=float, default=4e-1)

    # parser.add_argument('--cs', type=list,  default=[3, 32, 64, 128, 1024])
    parser.add_argument('--cs', type=list,  default=[3, 16, 32, 32, 1024])

    parser.add_argument("--add_sn", type=bool, default=False)

    parser.add_argument('--z0_dim', type=int, default=5, help='size of the latent z vector') # 100
    parser.add_argument('--z1_dim', type=int, default=5, help='size of the latent z vector') # 100
    parser.add_argument('--z2_dim', type=int, default=5, help='size of the latent z vector') # 100
    parser.add_argument('--z3_dim', type=int, default=40, help='size of the latent z vector') # 100

    parser.add_argument('--beta1E',  type=float, default=0., help='beta1 for adam. default=0.5')
    parser.add_argument('--beta1G',  type=float, default=0., help='beta1 for adam GI. default=0.5')
    parser.add_argument('--beta1I',  type=float, default=0., help='beta1 for adam GI. default=0.5')
    parser.add_argument('--e_decay', type=float, default=0.0000, help='weight decay for E')
    parser.add_argument('--i_decay', type=float, default=0.0005, help='weight decay for I')
    parser.add_argument('--g_decay', type=float, default=0.0005, help='weight decay for G')
    parser.add_argument('--e_gamma', type=float, default=0.998, help='lr decay for EBM')
    parser.add_argument('--i_gamma', type=float, default=0.998, help='lr decay for I')
    parser.add_argument('--g_gamma', type=float, default=0.998, help='lr decay for G')

    parser.add_argument('--fid', type=bool, default=True)
    parser.add_argument('--n_metrics', type=int, default=1) #10
    parser.add_argument('--n_metrics_start', type=int, default=20)

    parser.add_argument('--job_id', type=int, default=0)
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
    gpus_num = 2
    process_num = 1
    use_pygrid = True

    output_dir = './{}/'.format(os.path.splitext(os.path.basename(__file__))[0])
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    output_dir += t + '/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + 'code/', exist_ok=True)

    [save_file(output_dir, f) for f in
    ['config.py', 'dataset.py', 'nets.py', 'utils.py', 'pygrid.py', 'pygrid_utils.py',
     os.path.basename(__file__)]]

    if use_pygrid:
        import pygrid
        device_ids = [i for i in range(gpus_num)] * process_num
        workers = len(device_ids)

        # set devices
        pygrid.init_mp()
        pygrid.fill_queue(device_ids)

        # set opts
        get_opts_filename = lambda exp: output_dir + '{}.csv'.format(exp)
        exp_id = pygrid.get_exp_id(__file__)

        write_opts = lambda opts: pygrid.write_opts(opts, lambda: open(get_opts_filename(exp_id), mode='w', newline=''))
        read_opts = lambda: pygrid.read_opts(lambda: open(get_opts_filename(exp_id), mode='r'))

        if not os.path.exists(get_opts_filename(exp_id)):
            write_opts(create_args_grid())
        write_opts(pygrid.reset_job_status(read_opts()))

        # set logging
        logger = Logger(output_dir, 'main')
        logger.info(f'available devices {device_ids}')

        # run
        pygrid.run_jobs(logger, exp_id, output_dir, workers, letgo, read_opts, write_opts, update_job_result)
        logger.info('done')

    else:
        opt = dict()
        letgo(opt, output_dir, {})

if __name__ == '__main__':
    main()
