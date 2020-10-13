import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import random
import torch.optim as optim
import csv
import tqdm
from torch.nn import init

from Network.unet_model import UNet
from Network.squeeze_unet import Squeeze_UNet
from Network.mobile_unet import Mobile_UNet
from Network.shuffleV2_unet import ShuffleV2_UNet
from Network.shuffleV1_unet import ShuffleV1_UNet
from Network.igcV1_unet import IGCV1_UNet

from Loss.loss import LossBinary
from evaluation import *
from lookahead import Lookahead


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = LossBinary(0.3)

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.train_log = config.train_log
        self.valid_log = config.valid_log

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = list(map(int, config.cuda_idx.split(',')))
        self.model_type = config.model_type
        self.t = config.t

        self.build_model()

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def build_model(self):
        """Build generator and discriminator."""
        self.set_seed(1)

        if self.model_type == 'UNet':
            self.unet = UNet()
        elif self.model_type == 'Squeeze_UNet':
            self.unet = Squeeze_UNet()
        elif self.model_type == 'Mobile_UNet':
            self.unet = Mobile_UNet()
        elif self.model_type == 'ShuffleV1_UNet':
            self.unet = ShuffleV1_UNet()
        elif self.model_type == 'ShuffleV2_UNet':
            self.unet = ShuffleV2_UNet()
        elif self.model_type == 'IGCV1_UNet':
            self.unet = IGCV1_UNet()

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2], weight_decay=0.00005)
        self.optimizer = Lookahead(self.optimizer, k=5, alpha=0.5)

        self.unet = nn.DataParallel(self.unet, self.device).cpu()

        self.init_weights(self.unet, init_type='kaiming')

        self.print_network(self.unet, self.model_type)

    def init_weights(self, net, init_type='uniform', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'uniform':
                    init.kaiming_uniform_(m.weight, a=0, mode='fan_in')
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

        print("The number of parameters: {}".format(num_params))
        print(name)

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def cuda(self, x):
        return x.cuda(async=True) if torch.cuda.is_available() else x

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#

        unet_path = os.path.join(self.model_path, '%s.pt' % (self.model_type))
        lr = self.lr
        best_unet_score = 0.
        num_epochs = self.num_epochs
        # U-Net Train

        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            state = torch.load(str(unet_path), map_location='cpu')  # 加载整个网络模型
            epoch = state['epoch']
            step = state['step']
            self.unet.load_state_dict(state['model'])  # 加载网络参数
            print('Restored model, epoch {}, step {:,}'.format(epoch, step))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            # Train for Encoder
            epoch = 1
            step = 0

        save = lambda ep: torch.save({
            'model': self.unet.state_dict(),
            'epoch': ep,
            'step': step,
        }, str(unet_path))


        for epoch in range(epoch, num_epochs+1):

            self.unet.train(True)  # 训练状态 测试时用unet.eval()将DropOut和BatchNorm锁住
            losses = []

            SE = []  # Sensitivity (Recall)
            SP = []  # Specificity
            PPV = []
            JS = []  # Jaccard Similarity
            DC = []  # Dice Coefficient
            length = 0

            try:
                epoch_loss = 0
                mean_loss = 0
                tq = tqdm.tqdm(total=(len(self.train_loader) * self.batch_size))  # train_loader是以batch_size为单位
                tq.set_description('Epoch %d, lr %0.5f' % (epoch, lr))

                for i, (images, GT) in enumerate(self.train_loader):
                    # GT : Ground Truth

                    images = self.cuda(images)
                    GT = self.cuda(GT)

                    # SR : Segmentation Result
                    SR = self.unet(images)

                    loss = self.criterion(SR, GT)

                    epoch_loss += loss.item() * images.size(0)
                    losses.append(loss.item())

                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()
                    step += 1

                    mean_loss = np.mean(losses)

                    PPV += get_ppv(SR, GT, 0.3)
                    SE += get_sen(SR, GT, 0.3)
                    SP += get_spe(SR, GT, 0.3)
                    JS += get_jaccard(SR, GT, 0.3)
                    DC += get_dice(SR, GT, 0.3)

                    length += images.size(0)
                    tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                    tq.update(images.size(0))

                tq.close()

                train_SE = np.mean(SE).astype(np.float64)
                train_SP = np.mean(SP).astype(np.float64)
                train_PPV = np.mean(PPV).astype(np.float64)
                train_JS = np.mean(JS).astype(np.float64)
                train_DC = np.mean(DC).astype(np.float64)
                train_epoch_loss = epoch_loss / length

                # Print the log info
                print('Epoch [%d/%d], Loss: %.4f, \n[Training] SE: %.4f, SP: %.4f,  JS: %.4f, DC: %.4f, PPV: %.4f' % (
                    epoch, self.num_epochs, \
                    train_epoch_loss, \
                    train_SE, train_SP, train_JS, train_DC, train_PPV))

                f = open(os.path.join(self.train_log, 'log.csv'), 'a', encoding='utf-8', newline='')
                wr = csv.writer(f)
                wr.writerow(
                    [self.model_type, 'SE', train_SE, 'SP', train_SP, 'JS', train_JS, 'DC', train_DC, 'PPV', train_PPV,
                     'lr', self.lr, 'epoch', self.num_epochs, 'decay', self.num_epochs_decay, ])

                f.close()

                # Decay learning rate
                if (epoch + 1) > (self.num_epochs_decay):
                    lr -= (self.lr / self.num_epochs_decay)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print('Decay learning rate to lr: {}.'.format(lr))

                # ===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()

                SE = []  # Sensitivity (Recall)
                SP = []  # Specificity
                PPV = []
                JS = []  # Jaccard Similarity
                DC = []  # Dice Coefficient
                length = 0
                losses = 0
                with torch.no_grad():
                    for i, (images, GT) in enumerate(self.valid_loader):
                        images = self.cuda(images)
                        GT = self.cuda(GT)
                        SR = self.unet(images)
                        loss = self.criterion(SR, GT)

                        losses += loss.item() * images.size(0)
                        PPV += get_ppv(SR, GT, 0.3)
                        SE += get_sen(SR, GT, 0.3)
                        SP += get_spe(SR, GT, 0.3)
                        JS += get_jaccard(SR, GT, 0.3)
                        DC += get_dice(SR, GT, 0.3)

                        length += images.size(0)

                val_SE = np.mean(SE).astype(np.float64)
                val_SP = np.mean(SP).astype(np.float64)
                val_PPV = np.mean(PPV).astype(np.float64)
                val_JS = np.mean(JS).astype(np.float64)
                val_DC = np.mean(DC).astype(np.float64)
                val_mean_loss = losses / length
                unet_score = val_JS + val_DC

                print('[Validation] Loss: %.4f , SE: %.4f, SP: %.4f, JS: %.4f, DC: %.4f, PPV: %.4f' % (
                    val_mean_loss, val_SE, val_SP, val_JS, val_DC, val_PPV))

                save_SR = (SR >= 0.5).float()
                # save_SR = SR
                torchvision.utils.save_image(save_SR.data.cpu(),
                                             os.path.join(self.valid_log,
                                                          '%s_valid_%d_SR.png' % (self.model_type, epoch)))
                torchvision.utils.save_image(images.data.cpu(),
                                             os.path.join(self.valid_log,
                                                          '%s_valid_%d_images.png' % (self.model_type, epoch)))

                torchvision.utils.save_image(GT.data.cpu(),
                                             os.path.join(self.valid_log,
                                                          '%s_valid_%d_GT.png' % (self.model_type, epoch)))

                # Save Best U-Net model
                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    save_folder = './models/best_' + self.model_type + "_model/"
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    save_path = save_folder + self.model_type + '-' + str(best_unet_score) + '-' + str(
                        val_JS) + '-' + str(val_DC) + '.pt'

                    torch.save({
                        'model': best_unet,
                        'epoch': epoch,
                        'step': step,
                    }, save_path)
                    print('Best %s model score : %.4f' % (self.model_type, best_unet_score))

                f = open(os.path.join(self.valid_log, 'log.csv'), 'a', encoding='utf-8', newline='')
                wr = csv.writer(f)
                wr.writerow(
                    [self.model_type, 'SE', val_SE, 'SP', val_SP, 'JS', val_JS, 'DC', val_DC, 'PPV', val_PPV, 'lr',
                     self.lr, 'epoch', self.num_epochs, 'decay', self.num_epochs_decay])
                f.close()
                save(epoch + 1)
            except KeyboardInterrupt:
                tq.close()
                save(epoch)
                print('done')
                return

    def test(self):
        # ===================================== Test ====================================#
        best_unet_path = os.path.join(self.model_path, ('best_' + self.model_type + '_model'))
        best_unet = os.listdir(best_unet_path)[-1]
        unet_path = os.path.join(best_unet_path, str(best_unet))

        # unet_path = os.path.join(self.model_path, '%s.pt' % (self.model_type))
        state = torch.load(str(unet_path), map_location='cpu')
        self.unet.load_state_dict(state['model'])

        SE = []  # Sensitivity (Recall)
        SP = []  # Specificity
        PPV = []
        JS = []  # Jaccard Similarity
        DC = []  # Dice Coefficient
        with torch.no_grad():  # 测试时不求导
            for i, (images, GT) in enumerate(self.test_loader):
                images = self.cuda(images)
                GT = self.cuda(GT)
                SR = self.unet(images)

                PPV += get_ppv(SR, GT, 0.3)
                SE += get_sen(SR, GT, 0.3)
                SP += get_spe(SR, GT, 0.3)
                JS += get_jaccard(SR, GT, 0.3)
                DC += get_dice(SR, GT, 0.3)


        SE = np.mean(SE).astype(np.float64)
        SP = np.mean(SP).astype(np.float64)
        PPV = np.mean(PPV).astype(np.float64)
        JS = np.mean(JS).astype(np.float64)
        DC = np.mean(DC).astype(np.float64)

        save_SR = (SR > 0.5).float()
        # save_SR = SR
        torchvision.utils.save_image(save_SR.data.cpu(),
                                     os.path.join(self.result_path, ('%s_test_SR.png'%self.model_type)))
        torchvision.utils.save_image(GT.data.cpu(),
                                     os.path.join(self.result_path, ('%s_test_GT.png'%self.model_type)))
        torchvision.utils.save_image(images.data.cpu(),
                                     os.path.join(self.result_path, ('%s_test_image.png'%self.model_type)))

        f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow(
            [self.model_type, 'SE', SE, 'SP', SP, 'JS', JS, 'DC', DC, 'PPV', PPV])
        f.close()
