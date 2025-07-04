import os
import yaml
from optparse import OptionParser
config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
base_path = config['base_path']
base_path = os.path.normpath(base_path)
res_path = config['res_path']
res_path = os.path.normpath(res_path)
# make dir if not exist
if not os.path.exists(res_path):
    os.makedirs(res_path)

parser = OptionParser()
# device
parser.add_option('--gpu', type='str', dest='gpu',
                  default='cuda:3')

parser.add_option('--train_data_path', type='str', dest='train_data_path',
                    default=os.path.join(base_path, 'train'))
parser.add_option('--train_label_path', type='str', dest='train_label_path',
                    default=os.path.join(base_path, 'train_mask'))
parser.add_option('--val_data_path', type='str', dest='val_data_path',  
                    default=os.path.join(base_path, 'val'))
parser.add_option('--val_label_path', type='str', dest='val_label_path',
                    default=os.path.join(base_path, 'val_mask'))

# parser.add_option('--train_data_path', type='str', dest='train_data_path',
#                   default=r'/mnt/no1/liuguannan/Brain/net_data_Tif/train')
# parser.add_option('--train_label_path', type='str', dest='train_label_path',
#                   default=r'/mnt/no1/liuguannan/Brain/net_data_Tif/train_mask')
# parser.add_option('--val_data_path', type='str', dest='val_data_path',
#                   default=r'/mnt/no1/liuguannan/Brain/net_data_Tif/val')
# parser.add_option('--val_label_path', type='str', dest='val_label_path',
#                   default=r'/mnt/no1/liuguannan/Brain/net_data_Tif/val_mask')

# data_properity
parser.add_option('-p', '--patch_size', dest='patch_size', default=[256, 256] , type='int',
                  help='patch size')  # w h d
parser.add_option('-b', '--batch_size', dest='batch_size', default=4,
                  type='int', help='batch size')
parser.add_option('--input_channel', dest='input', default=1,
                  type='int', help='input channel')
parser.add_option('--output_channel', dest='output', default=1, type='int', help='number of classes')


# dataLoader
parser.add_option('--pin_memory', dest='pin_memory', default=True, help='pin_memory')
parser.add_option('-w', '--num_workers', dest='num_workers', default=0, type='int',
                  help='multi-preprocess num workers')
# train
parser.add_option('-e', '--epochs', dest='epochs', default=150, type='int',
                  help='number of epochs')
parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                  type='float', help='learning rate')
parser.add_option('-r', '--resume', type='str', dest='load', default=False,
                  help='load pretrained model')
parser.add_option('--rlt', type='float', dest='rlt', default=0.2, help='relation between CE/FL and dice')
parser.add_option('--norm', type='str', dest='norm', default='bn')
# save
# parser.add_option('--result_path', type='str', dest='res_path',
#                   default='/mnt/no1/liuguannan/Brain/result', help='result path')

parser.add_option('--result_path', type='str', dest='res_path',
                    default=res_path, help='result path')

(options, args) = parser.parse_args()
