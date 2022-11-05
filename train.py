import argparse
from src.dataset import get_train_val_loader
from src.config import get_device, get_model, get_criterion, get_optimizer, get_scheduler
from src.trainer import Trainer
from src.utils import fixed_random_seed, save_json
from src.constant import RANDOM_SEED, CONFIG_PATH


@fixed_random_seed(RANDOM_SEED)
def train(args):
    train_loader, val_loader = get_train_val_loader(args)
    device = get_device(args.device[0])
    model = get_model(args.model, args.num_classes)
    criterion = get_criterion(args.loss)
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_scheduler(args, optimizer)

    trainer = Trainer(
        model, device,
        train_loader, val_loader,
        criterion, optimizer, lr_scheduler, args)

    trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1,
                        help='fold')
    parser.add_argument('-ep', '--epoch', type=int, default=200,
                        help='epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('-agbs', '--accumulate_grad_bs', type=int, default=1,
                        help='accumulate gradient batches')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        help='model')
    parser.add_argument('-cls', '--num_classes', type=int, default=33,
                        help='number of classes')
    parser.add_argument('--image_size', type=int, default=512,
                        help='crop and resize to img_size')

    # set optimization
    parser.add_argument('--loss', type=str, default='FL',
                        help='loss function')
    parser.add_argument('--optim', type=str, default='AdamW',
                        help='optimizer')
    parser.add_argument('--lr', type=float, default=3e-4,
                         help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--scheduler', type=str, default='step',
                        help='learning rate schedule')
    parser.add_argument('--step_size', type=int, default=2000,
                        help='learning rate decay period')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='learning rate decay factor')

    # augmentation
    parser.add_argument('--autoaug', type=float, default=0,
                        help='probability of auto-augmentation')

    # set device
    parser.add_argument('--device', type=int, default=[0], nargs='+',
                        help='index of gpu device')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='numbers of workers')

    args = parser.parse_args()
    save_json(CONFIG_PATH, args)
    train(args)
