import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random



def main(config):
    # cudnn.benchmark = True # 大部分情况下，设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    # if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
    #     print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    #     print('Your input for model_type was %s'%config.model_type)
    #     return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    config.train_log = os.path.join(config.train_log, config.model_type)
    if not os.path.exists(config.train_log):
        os.makedirs(config.train_log)

    config.valid_log = os.path.join(config.valid_log, config.model_type)
    if not os.path.exists(config.valid_log):
        os.makedirs(config.valid_log)

    decay_ratio = 0.8
    decay_epoch = int(config.num_epochs * decay_ratio)
    config.num_epochs_decay = decay_epoch

    print(config)

    train_loader = get_loader(image_path=config.train_path,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              shuffle=True
                              )
    valid_loader = get_loader(image_path=config.valid_path,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid',
                              shuffle=True
                              )
    test_loader = get_loader(image_path=config.test_path,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             )

    solver = Solver(config, train_loader, valid_loader, test_loader)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    else:
        solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=192)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=70)
    parser.add_argument('--num_epochs_decay', type=int, default=10) #衰减 当num_epoch大于num_epoch_decay 动态调整学习率
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.9)  # momentum1 in          Adam 优化器参数
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam


    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='Squeeze_UNet',
                        help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/best_U_Net/UNet_base/Reccurent_GRU_UNet/unet_2D/Reccurent_GRU_SE_UNet/BAM_UNet/CBAM_UNet')
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--train_path', type=str, default='./data/train/images/')
    parser.add_argument('--valid_path', type=str, default='./data/val/images/')
    parser.add_argument('--test_path', type=str, default='./data/test/images/')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--train_log', type=str, default='./train_log/')
    parser.add_argument('--valid_log', type=str, default='./valid_log/')
    parser.add_argument('--cuda_idx', type=str, default='0')

    config = parser.parse_args()
    main(config)
