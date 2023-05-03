import os
import os.path as osp
import logging
import gc
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import MinkowskiEngine as ME

from model_colorization import load_model
from lib.timer import Timer, AverageMeter

def colorfulness_metric(points):
    """
    :input point: (N*6, xyz, rgb)
    :return: metric of point cloud colors
    """
    assert points.shape[1] ==3, 'wrong pcd shape'
    rg = np.abs(points[:, 0] - points[:, 1])
    yb = np.abs(0.5*(points[:, 0] + points[:, 1]) - points[:, 2])

    rgMean, rgStd = np.mean(rg), np.std(rg)
    ybMean, ybStd = np.mean(yb), np.std(yb)

    stdRoot = np.sqrt((rgStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rgMean ** 2) + (ybMean ** 2))

    return stdRoot + (0.3 * meanRoot)


class STrainer:
    def __init__(
            self,
            config,
            data_loader=None):
        num_feats = 36

        # Model initialization
        Model = load_model(config.model)
        model = Model(
            num_feats,
            config.model_n_out,
            bn_momentum=config.bn_momentum,
            normalize_feature=config.normalize_feature,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3)

        # load pretrained model weights
        if config.weights:
            checkpoint = torch.load(config.weights)
            model.load_state_dict(checkpoint['state_dict'])

        # set default hyper parameters
        self.config = config
        self.model = model
        self.max_epoch = config.max_epoch
        self.save_freq = config.save_freq_epoch

        if config.use_gpu and not torch.cuda.is_available():
            logging.warning('Warning: There\'s no CUDA support on this machine, '
                            'training is performed on CPU.')
            raise ValueError('GPU not available, but cuda flag set')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.optimizer = getattr(optim, config.optimizer)(
            model.parameters(),
            lr=config.lr)

        self.start_epoch = 1
        self.checkpoint_dir = config.out_dir
        self.iter_size = config.iter_size
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader

        #ensure_dir(self.checkpoint_dir)
        self.model = self.model.to(self.device)
        if config.resume is not None:
            if osp.isfile(config.resume):
                logging.info("=> loading checkpoint '{}'".format(config.resume))
                state = torch.load(config.resume)
                self.start_epoch = state['epoch']
                model.load_state_dict(state['state_dict'])
                self.optimizer.load_state_dict(state['optimizer'])

            else:
                raise ValueError(f"=> no checkpoint found at '{config.resume}'")

    def train(self):
        for epoch in range(self.start_epoch, self.max_epoch + 1):
            lr = self.config.lr
            #logging.info(f" Epoch: {epoch}, LR: {lr}")
            self._train_epoch(epoch)
            self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0

        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()

        iter_size = self.iter_size
        #start_iter = (epoch - 1) * (len(data_loader) // iter_size)
        #criterion = nn.MSELoss(reduction='mean')
        criterion = nn.SmoothL1Loss()
        #criterion = nn.CrossEntropyLoss(ignore_index=-1)
        data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
        # Training steps
        for curr_iter in range(len(data_loader) // iter_size): #0-5
            #print("#########################")
            self.optimizer.zero_grad()
            batch_loss = 0
            # records data loading time
            data_time = 0
            total_timer.tic()
            for iter_idx in range(iter_size): #1
                data_timer.tic()
                input_dict = data_loader_iter.next()
                data_time += data_timer.toc(average=False)

                # extract feature from network
                sinput = ME.SparseTensor(input_dict['sinput_F'],coords=input_dict['sinput_C']).to(self.device)
                F0 = self.model(sinput).F
                #loss = criterion(F0, input_dict['rgb'].cuda())
                #loss2 = colorfulness_metric(F0.cpu().detach().numpy())
                #loss = criterion(F0, input_dict['rgb'].cuda())
                loss1 = criterion(F0[:, 0:3], input_dict['rgb'].cuda())
                loss2 = criterion(F0[:, 3:6], input_dict['rgb'].cuda())
                loss3 = criterion(F0[:, 6:9], input_dict['rgb'].cuda())
                temp = torch.min(loss1, loss2)
                loss = torch.min(temp, loss3) + 0.08 * loss1 + 0.04 * loss2 + 0.02 * loss3
                loss /= iter_size
                loss.backward()
                batch_loss += loss.item()

            self.optimizer.step()
            gc.collect()
            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            if curr_iter % self.config.stat_freq == 0:
                #self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
                print("Train Epoch: {} [{}/{}], Current Loss: {:.3e}".format(epoch, curr_iter, len(self.data_loader) //
                            iter_size, batch_loss) + "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                        data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
                data_meter.reset()
                total_timer.reset()


    def _save_checkpoint(self, epoch, filename='checkpoint'):
        if epoch % self.save_freq == 0:
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'config': self.config,
            }
            filename = str(epoch) + '_checkpoint'
            filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
            logging.info("Saving checkpoint: {} ...".format(filename))
            torch.save(state, filename)


