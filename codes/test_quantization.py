import argparse
from collections import OrderedDict
import time

import torch
from utils.util import seg_add_pad, seg_remove_pad

import utils.util as util
import options.options as option
from data import create_dataloader, create_dataset
from models.modules.Quantization_h265_rgb_stream import \
    Quantization_H265_Stream


def h265_compress(data, quantization_H265_Stream, Seg_Len = 5, frame_skip = True, device = torch.device('cpu')):
    input = data['GT']
    b, c, bt, h, w = input.size()
    input_reshaped = input.permute(0, 2, 1, 3, 4)
    out_video, pad_num = seg_add_pad(input_reshaped, Seg_Len)
    b, seg_num, seg_len, c, h, w = out_video.size()

    quantization_H265_Stream.open_writer(device, w, h)
    for seg_i in range(seg_num):
        if seg_i != seg_num - 1 and frame_skip:
            c, h, w  = out_video.size(-3),out_video.size(-2),out_video.size(-1)
            out = out_video[:,seg_i].reshape(-1,c,h,w)
            out_next = out_video[:,seg_i+1].reshape(-1,c,h,w)
            out[2] = out[3]
            out[3] = out[4]
            out[4] = out_next[0]
            quantization_H265_Stream.write_multi_frames(out)
        else:
            c, h, w  = out_video.size(-3),out_video.size(-2),out_video.size(-1)
            out = out_video[:,seg_i].reshape(-1,c,h,w)
            quantization_H265_Stream.write_multi_frames(out)
    
    bpp = quantization_H265_Stream.close_writer()

    quantization_H265_Stream.open_reader()
    outs = []
    for seg_i in range(seg_num):
        v_seg = quantization_H265_Stream.read_multi_frames(Seg_Len)
        outs += [v_seg]
    
    out = torch.cat(outs,dim=0)
    h, w = out.size(-2),out.size(-1)
    out = out.reshape(b, -1, Seg_Len, 3, h, w)
    out = seg_remove_pad(out,pad_num,Seg_Len)

    regenerate = out.reshape(-1, 3, h, w)
    
    GT = input_reshaped.squeeze()

    return regenerate, GT, bpp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        print('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    quantization_H265_Stream = Quantization_H265_Stream(9, -1, 2, opt)
            
    for test_loader in test_loaders:
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['img_bpp'] = []
        
        test_set_name = test_loader.dataset.opt['name']
        print('\nTesting [{:s}]...'.format(test_set_name))

        for data in test_loader:
            regenerate, GT, bpp = h265_compress(data, quantization_H265_Stream)
            test_results["img_bpp"] += [bpp]
            
            psnr = util.calculate_psnr(regenerate, GT)
            ssim = util.calculate_ms_ssim(regenerate, GT)
            test_results['psnr'] += psnr 
            test_results['ssim'] += ssim
        
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print(
                '----Average PSNR/SSIM results for {}----\n\tpsnr: {:.6f} db; ssim: {:.6f}. \n'.format(
                test_set_name, ave_psnr, ave_ssim))

        ave_img_bpp = sum(test_results['img_bpp']) / len(test_results['img_bpp'])
        ave_img_bpp = ave_img_bpp.mean()
        print(
                '----Average Compression results for {}----\n\t ave_img_bpp: {:.6f}dB.\n'.format(
                test_set_name, ave_img_bpp))

