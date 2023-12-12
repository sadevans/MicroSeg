import yaml
import argparse


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='scunet_color_real_psnr', help='scunet_color_real_psnr, scunet_color_real_gan')
    parser.add_argument('--test_dataset_name', type=str, default='real3', help='test set, bsd68 | set12 | G-123 | G46 | b2u_g123_stride200')
    parser.add_argument('--show_img', type=bool, default=False, help='show the image')
    parser.add_argument('--model_zoo', type=str, default='./model_zoo', help='path of model_zoo')
    parser.add_argument('--test_dir', type=str, default='./data/test', help='path of testing folder')
    parser.add_argument('--results', type=str, default='./results', help='path of results')
    # parser.add_argument('--results', type=str, default='./data/test', help='path of results')


    return parser