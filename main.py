import argparse
# from train import train
from test import test
import os, random
import torch
import datetime

import os

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


from tensorboardX import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from sampler import InfiniteSamplerWrapper

from net import Net
from datasets import TrainDataset, sample_data
from util import adjust_learning_rate

cudnn.benchmark = True


def train(rank, *args):
    args = args[0]
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d' % args.port, world_size=torch.cuda.device_count(), rank=rank, group_name='mtorch')

    # Device, save and log configuration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rank==0:
        save_dir = Path(os.path.join(args.save_dir, args.name))
        save_dir.mkdir(exist_ok=True, parents=True)
        log_dir = Path(os.path.join(args.log_dir, args.name))
        log_dir.mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(log_dir=str(log_dir))

    # Prepare datasets

    content_dataset = TrainDataset(args.content_dir, args.img_size)
    texture_dataset = TrainDataset(args.texture_dir, args.img_size, gray_only=True)
    color_dataset = TrainDataset(args.color_dir, args.img_size)

    sampler = DistributedSampler(content_dataset, shuffle=True)
    content_iter = DataLoader(content_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, drop_last=True, pin_memory=False, persistent_workers=True)
    
    content_iter = sample_data(content_iter, sampler)

    sampler = DistributedSampler(texture_dataset, shuffle=True)
    texture_iter = DataLoader(texture_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, drop_last=True, pin_memory=False, persistent_workers=True)
    texture_iter = sample_data(texture_iter, sampler)

    sampler = DistributedSampler(color_dataset, shuffle=True)
    color_iter = DataLoader(color_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, drop_last=True, pin_memory=False, persistent_workers=True)
    color_iter = sample_data(color_iter, sampler)

    # content_iter = iter(data.DataLoader(
    #     content_dataset, batch_size=args.batch_size,
    #     sampler=InfiniteSamplerWrapper(content_dataset),
    #     num_workers=args.n_threads))
    # texture_iter = iter(data.DataLoader(
    #     texture_dataset, batch_size=args.batch_size,
    #     sampler=InfiniteSamplerWrapper(texture_dataset),
    #     num_workers=args.n_threads))
    # color_iter = iter(data.DataLoader(
    #     color_dataset, batch_size=args.batch_size,
    #     sampler=InfiniteSamplerWrapper(color_dataset),
    #     num_workers=args.n_threads))

    # Prepare network

    network = Net(args)
    network.train()
    network.to(device)
    network = DistributedDataParallel(network, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    # Training options
    network = network.module
    opt_L = torch.optim.Adam(network.L_path.parameters(), lr=args.lr)
    opt_AB = torch.optim.Adam(network.AB_path.parameters(), lr=args.lr)

    opts = [opt_L, opt_AB]

    # Start Training
    stime = datetime.datetime.now()

    for i in range(args.max_iter):
        # S1: Adjust lr and prepare data

        adjust_learning_rate(opts, iteration_count=i, args=args)

        content_l, content_ab = [x.to(device) for x in next(content_iter)]
        texture_l = next(texture_iter).to(device)
        color_l, color_ab = [x.to(device) for x in next(color_iter)]

        # S2: Forward

        l_pred, ab_pred = network(content_l, content_ab, texture_l, color_ab)

        # S3: Calculate loss

        loss_ct, loss_t = network.ct_t_loss(l_pred, content_l, texture_l)
        loss_cr = network.cr_loss(ab_pred, color_ab)

        loss_ctw = args.content_weight * loss_ct
        loss_tw = args.texture_weight * loss_t
        loss_crw = args.color_weight * loss_cr

        loss = loss_ctw + loss_tw + loss_crw

        # S4: Backward

        for opt in opts:
            opt.zero_grad()
        loss.backward()
        for opt in opts:
            opt.step()

        # S5: Summary loss and save subnets
        if rank==0:
            ctime = datetime.datetime.now()
            duration = (ctime - stime)  * (args.max_iter - i)
            duration = duration.total_seconds() / 3600.0
        
            print('[(+%.1fh)] Init Iter %08d loss_content=%.6f loss_texture=%.6f loss_color=%.6f'%(duration, i,loss_ct.item(),loss_t.item(),loss_cr.item()  ))
            stime = ctime
            
            writer.add_scalar('loss_content', loss_ct.item(), i + 1)
            writer.add_scalar('loss_texture', loss_t.item(), i + 1)
            writer.add_scalar('loss_color', loss_cr.item(), i + 1)

            if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
                state_dict = network.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))
                torch.save(state_dict, save_dir /
                        'network_iter_{:d}.pth.tar'.format(i + 1))
    if rank==0:
        writer.close()

def Args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--mode', type=str, default='train',
                        help='Train or test')
    parser.add_argument('--data', type=str, default='./',
                        help='data root path')
    parser.add_argument('--content_dir', type=str, default="content.txt",
                        help='Directory path to a batch of content images')
    parser.add_argument('--texture_dir', type=str, default="texture.txt",
                        help='Directory path to a batch of texture images')
    parser.add_argument('--color_dir', type=str, default="color.txt",
                        help='Directory path to a batch of Color images')
    parser.add_argument('--out_root', type=str, default='output/',
                        help='Root directory for outputs')
    parser.add_argument('--network', type=str, default='models/net_final.pth')
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
    parser.add_argument('--name', default='AAST',
                        help='Name of this model')


    # training options
    parser.add_argument('--save_dir', default='./checkpoints',
                        help='Directory to save the checkpoints')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--img_size', default=256, type=int,
                        help='Size of input img')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--texture_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--color_weight', type=float, default=10.0)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    
    # test options
    parser.add_argument('--test_opt', type=str, default='TC',
                        help='Test options, ST(Style Transfer), T(Texture Only), C(Color Only), TC(Texture and Color), INT(Interpolation)')
    parser.add_argument('--int_num', type=int, default=4,
                        help='Interpolation num')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = Args()
    args.content_dir = os.path.join(args.data, args.content_dir)
    args.texture_dir = os.path.join(args.data, args.texture_dir)
    args.color_dir = os.path.join(args.data, args.color_dir)
    args.save_dir = os.path.join(args.data, args.save_dir)
    # if args.mode == 'train':
    #     train(args)
    # elif args.mode == 'test':
    #     test(args)

    args.port = random.randint(23400, 23499)
    args.batch_size = args.batch_size // torch.cuda.device_count()

    torch.multiprocessing.spawn(train, nprocs=torch.cuda.device_count(), args=(args,) )