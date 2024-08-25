import argparse


parser = argparse.ArgumentParser(description="Image Classification Toolbox")

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')
parser.add_argument("--device", type=str, default='cuda', help='cpu or cuda')

# Data specifications
parser.add_argument('--data_name', type=str, default='SoybeanSeed', help='dataset name: Cassva, SoybeanSeed, ')
parser.add_argument('--data_root', type=str, default='./', help='dataset directory')
parser.add_argument("--img_size", type=int, default=512, help='image size')
parser.add_argument("--fold_num", type=int, default=5, help='class numble')

# Model specifications
parser.add_argument('--outf', type=str, default=None, help='saving_path')
parser.add_argument('--method', type=str, default='SwinTransformer', help='method name：Efficientnet, Vit, SwinTransformer, AlexNet, MLPMixer, Timm_Efficientnet, Timm_Vit')
parser.add_argument('--pretrained', action='store_true', help='use pretrained ckpt')
# parser.add_argument('--pretrained', type=bool, default=True, help='use pretrained ckpt')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')

# Training specifications
parser.add_argument('--batch_size', type=int, default=10, help='the number of images per batch')
parser.add_argument("--max_epoch", type=int, default=20, help='total epoch')
parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--learning_rate", type=float, default=0.0001)

opt = parser.parse_args()

# Saving specifications
if opt.outf is None:
    opt.outf = './exp/' + str(opt.method)

# dataset path
if opt.data_name.find('Cassva') >= 0:
    opt.train_img_path = f"{opt.data_root}/data/train_images/"
    opt.train_csv_path = f"{opt.data_root}/data/train.csv"
    opt.test_img_path = None
    opt.n_class = 5
elif opt.data_name.find('SoybeanSeed') >= 0:
    opt.train_img_path = f"{opt.data_root}/data/Soybean_Seeds/"
    opt.train_csv_path = f"{opt.data_root}/data/Soybean_Seeds/train.csv"
    opt.test_img_path = None
    opt.n_class = 5
    if opt.img_size >= 227:
        opt.img_size = 227

if opt.method.find('Vit') >= 0 or opt.method.find('Timm_Vit') >= 0 or opt.method.find('SwinTransformer') >= 0 or opt.method.find('MLPMixer') >= 0:
    opt.img_size = 224

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False