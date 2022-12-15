import argparse
from src.dataset import get_train_val_loader
from src.builder import get_device, get_train_model, get_criterion, get_optimizer, get_scheduler
from src.trainer import Trainer
from src.utils import set_random_seed, save_json
from src.constant import RANDOM_SEED, CONFIG_PATH


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--fold', type=int, default=0,
                        help='fold')
    parser.add_argument('-ep', '--epoch', type=int, default=200,
                        help='epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--train_num', type=int, default=-1,
                        help='number of training data')
    parser.add_argument('--val_num', type=int, default=-1,
                        help='number of validation data')
    parser.add_argument('-agbs', '--accum_grad_bs', type=int, default=1,
                        help='accumulate gradient batches')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        help='model')
    parser.add_argument('-cls', '--num_classes', type=int, default=33,
                        help='number of classes')
    parser.add_argument('--image_size', type=int, default=512,
                        help='crop and resize to img_size')
    parser.add_argument('--trans', type=str, default='v1',
                        help='data transforms')

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

    return parser.parse_args()


@set_random_seed(RANDOM_SEED)
def train(args):
    train_loader, val_loader = get_train_val_loader(args)
    device = get_device(args.device)

    model = get_train_model(
        model=args.model,
        num_classes=args.num_classes,
        device_ids=args.device)

    criterion = get_criterion(loss=args.loss)
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_scheduler(args, optimizer)

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        accum_grad_bs=args.accum_grad_bs,
        lr_scheduler=lr_scheduler
        )

    trainer.fit(epoch=args.epoch)


if __name__ == '__main__':
    args = parse_arguments()
    save_json(CONFIG_PATH, args)
    train(args)
