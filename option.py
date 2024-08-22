import argparse


parser = argparse.ArgumentParser(description="Cassva Image Classification Toolbox")

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')
parser.add_argument("--device", type=str, default='cuda', help='cpu or cuda')

# Data specifications
parser.add_argument('--data_root', type=str, default='./', help='dataset directory')
parser.add_argument("--img_size", type=int, default=512, help='image size')
parser.add_argument("--fold_num", type=int, default=5, help='class numble')


# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/Vit', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='Vit', help='method name')
parser.add_argument('--pretrained', action='store_true', help='use Timm pretrained ckpt')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')

# Training specifications
parser.add_argument('--batch_size', type=int, default=10, help='the number of images per batch')
parser.add_argument("--max_epoch", type=int, default=20, help='total epoch')
parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--learning_rate", type=float, default=0.0001)

opt = parser.parse_args()

# dataset

opt.train_img_path = f"{opt.data_root}/data/train_images/"
opt.train_csv_path = f"{opt.data_root}/data/train.csv"
opt.test_img_path = f"{opt.data_root}/data/test_images/"

if opt.method.find('Vit') >= 0:
    opt.img_size = 224

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False