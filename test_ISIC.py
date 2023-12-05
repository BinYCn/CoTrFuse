import warnings
import cv2
import pandas as pd
from config import get_config
import argparse
from network.CoTrFuse import SwinUnet as Vit
from test_block_ISIC import test_mertric_here

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_test_path', type=str,
                    default='/',
                    help='imgs val data path.')
parser.add_argument('--labels_test_path', type=str,
                    default='',
                    help='labels val data path.')
parser.add_argument('--csv_dir_test', type=str,
                    default='',
                    help='labels val data path.')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', default=
'configs/swin_tiny_patch4_window7_224_lite.yaml')
parser.add_argument('--num_classes', '-t', default=2, type=int, )
parser.add_argument('--device', default='cuda', type=str, )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--checkpoint', type=str, default='checkpoint/', )
parser.add_argument('--save_name', type=str, default='CoTrFuse/ISIC', )
args = parser.parse_args()
config = get_config(args)

model_savedir = args.checkpoint + args.save_name + '/'
save_name = model_savedir + 'ckpt'

df_test = pd.read_csv(args.csv_dir_test)
test_imgs, test_masks = args.imgs_test_path, args.labels_test_path
test_imgs = [''.join([test_imgs, '/', i + '.png')]) for i in df_test['image_id']]
test_masks = [''.join([test_masks, '/', i + '_segmentation.png')]) for i in df_test['image_id']]
imgs_test = [cv2.imread(i)[:, :, ::-1] for i in test_imgs]
masks_test = [cv2.imread(i)[:, :, 0] for i in test_masks]

print('image done')


if __name__ == '__main__':
    model = Vit(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    dice, miou, pre, recall, f1_score, pa = test_mertric_here(model, imgs_test, masks_test, save_name)
    f = open(model_savedir + 'log1' + '.txt', "a")
    f.write('dice' + str(float(dice)) + '  _miou' + str(miou) +
            '  _pre' + str(pre) + '  _recall' + str(recall) +
            ' _f1_score' + str(f1_score) + ' _pa' + str(pa) + '\n')
    f.close()
    print('Done!')
