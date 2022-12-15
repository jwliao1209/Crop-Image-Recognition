import os
import torch
import argparse

from tqdm import tqdm

from src.dataset import get_test_loader
from src.builder import get_device, get_topk_models
from src.tta import TestTimeAug
from src.utils import get_ckpt_config_args, get_topk_ckpt, init_models, save_csv
from src.constant import LABEL_CATEGORY_MAP, PRED_DIR


@torch.no_grad()
def infer(args):
    print(f'Checkpoint: {args.checkpoint}')
    infer_loader = get_test_loader(args, test_type='public_and_private')
    device = get_device(device_id=0)
    models = get_topk_models(args, device)
    topk_ckpt = get_topk_ckpt(args.checkpoint, args.topk)
    models = init_models(models, topk_ckpt, device)
    tta = TestTimeAug(models, args.tta_transform)
    infer_bar = tqdm(infer_loader, desc=f'Inference')

    pred_list = []
    for data in infer_bar:
        image = data['image'].to(device)
        pred_class_idxes = tta.predict(image).cpu().numpy()

        for filepath, class_idx in zip(data['filepath'], pred_class_idxes):
            pred_label = LABEL_CATEGORY_MAP[class_idx]
            pred_list.append({
                'filename': os.path.basename(filepath),
                'label': pred_label
                })

    infer_bar.close()
    save_csv(os.path.join(PRED_DIR,
            f"{args.checkpoint[0]}_top{args.topk[0]}_submission.csv"), pred_list)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default=['10-28-11-15-57_size1080'],
                        nargs='+', help='weight path')
    parser.add_argument('--topk', type=int,
                        default=[1],
                        nargs='+', help='weight of topk accuracy')
    parser.add_argument('--tta_transform', type=str,
                        default=['identity', 'hflip', 'vflip', 'hvflip', 'vhflip'],
                        nargs='+', help='transformation of test time augumentation')

    args = get_ckpt_config_args(parser.parse_args())
    infer(args)
