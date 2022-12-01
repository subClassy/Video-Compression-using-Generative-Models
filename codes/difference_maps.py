import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

src_path = '/media/himank/SSD/SelfC/results/test_codec_uvg_bf/selfc_mm_rocks_big'
output_path = '/media/himank/SSD/SelfC/results/test_codec_uvg_bf/selfc_mm_rocks_big/diff_maps'

os.makedirs(output_path, exist_ok=True)

# src_path = '/media/himank/SSD/histo_matched/histo_matched_rgb'

image_files = glob.glob(src_path + "/*gt.jpg")

def get_difference(img_gt, img_recon):
    ch1, ch2, ch3 = img_gt[:,:,0], img_gt[:,:,1], img_gt[:,:,2]
    ch1_r, ch2_r, ch3_r = img_recon[:,:,0], img_recon[:,:,1], img_recon[:,:,2]

    hist_b = np.abs(np.array(ch1, dtype=np.float32) - np.array(ch1_r, dtype=np.float32))
    hist_g = np.abs(np.array(ch2, dtype=np.float32) - np.array(ch2_r, dtype=np.float32))
    hist_r = np.abs(np.array(ch3, dtype=np.float32) - np.array(ch3_r, dtype=np.float32))

    return hist_b, hist_g, hist_r

def total_variation_loss(img):
     h_img, w_img = img.shape
     tv_h = np.power(img[1:,:] - img[:-1,:], 2).sum()
     tv_w = np.power(img[:,1:] - img[:,:-1], 2).sum()
     return (tv_h + tv_w)/(h_img * w_img)

draw_plot = True

means_h265 = {
    'mean_b': [],
    'mean_g': [],
    'mean_r': [],
    'mean_h': [],
    'mean_s': [],
    'mean_v': [],
}


means_selfc = {
    'mean_b': [],
    'mean_g': [],
    'mean_r': [],
    'mean_h': [],
    'mean_s': [],
    'mean_v': [],
}

for seq_no, seq in enumerate(image_files):
    img_gt = cv2.imread(seq)
    img_gt = img_gt[:, 576:1536, :]
    img_h265 = cv2.imread(seq[:-6] + 'h265.jpg')
    img_h265 = img_h265[:, 576:1536, :]
    img_selfc = cv2.imread(seq[:-6] + 'selfc.jpg')
    img_selfc = img_selfc[:, 576:1536, :]
    
    hist_b, hist_g, hist_r = get_difference(img_gt, img_h265)
    mean_b_h265 = sum(sum(hist_b)) / (1536*2048)
    mean_g_h265 = sum(sum(hist_g)) / (1536*2048)
    mean_r_h265 = sum(sum(hist_r)) / (1536*2048)

    if draw_plot:
        fig, axs = plt.subplots(2, 3, figsize=(24, 12))
    
        axs[0, 0].set_title("GT vs H265 'R-Channel'")
        axs[0, 0].imshow(hist_r)
        axs[0, 1].set_title("GT vs H265 'G-Channel'")
        axs[0, 1].imshow(hist_g)
        axs[0, 2].set_title("GT vs H265 'B-Channel'")
        axs[0, 2].imshow(hist_b)

    hist_b, hist_g, hist_r = get_difference(img_gt, img_selfc)
    mean_b_selfc = sum(sum(hist_b)) / (1536*2048)
    mean_g_selfc = sum(sum(hist_g)) / (1536*2048)
    mean_r_selfc = sum(sum(hist_r)) / (1536*2048)
    
    if draw_plot:
        axs[1, 0].set_title("GT vs SelfC 'R-Channel'")
        axs[1, 0].imshow(hist_r)
        axs[1, 1].set_title("GT vs SelfC 'G-Channel'")
        axs[1, 1].imshow(hist_g)
        axs[1, 2].set_title("GT vs SelfC 'B-Channel'")
        axs[1, 2].imshow(hist_b)

        # plt.savefig(output_path + '/' + seq.split("/")[-1][:-6] + '_rgb_diff.png')
        plt.show()
        plt.close()

    
    img2_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2HSV)
    img2_h265 = cv2.cvtColor(img_h265, cv2.COLOR_BGR2HSV)
    img2_selfc = cv2.cvtColor(img_selfc, cv2.COLOR_BGR2HSV)

    hist_h, hist_s, hist_v = get_difference(img2_gt, img2_h265)
    mean_h_h265 = sum(sum(hist_h)) / (1536*2048)
    mean_s_h265 = sum(sum(hist_s)) / (1536*2048)
    mean_v_h265 = sum(sum(hist_v)) / (1536*2048)

    if draw_plot:
        fig, axs = plt.subplots(2, 3, figsize=(24, 12))

        axs[0, 0].set_title("GT vs H265 'H-Channel'")
        axs[0, 0].imshow(hist_h)
        axs[0, 1].set_title("GT vs H265 'S-Channel'")
        axs[0, 1].imshow(hist_s)
        axs[0, 2].set_title("GT vs H265 'V-Channel'")
        axs[0, 2].imshow(hist_v)

    hist_h, hist_s, hist_v = get_difference(img2_gt, img2_selfc)
    
    mean_h_selfc = sum(sum(hist_h)) / (1536*2048)
    mean_s_selfc = sum(sum(hist_s)) / (1536*2048)
    mean_v_selfc = sum(sum(hist_v)) / (1536*2048)

    means_h265["mean_b"].append(mean_b_h265)
    means_h265["mean_g"].append(mean_g_h265)
    means_h265["mean_r"].append(mean_r_h265)
    means_h265["mean_h"].append(mean_h_h265)
    means_h265["mean_s"].append(mean_s_h265)
    means_h265["mean_v"].append(mean_v_h265)

    means_selfc["mean_b"].append(mean_b_selfc)
    means_selfc["mean_g"].append(mean_g_selfc)
    means_selfc["mean_r"].append(mean_r_selfc)
    means_selfc["mean_h"].append(mean_h_selfc)
    means_selfc["mean_s"].append(mean_s_selfc)
    means_selfc["mean_v"].append(mean_v_selfc)

    if draw_plot:
        axs[1, 0].set_title("GT vs SelfC 'H-Channel'")
        axs[1, 0].imshow(hist_h)
        axs[1, 1].set_title("GT vs SelfC 'S-Channel'")
        axs[1, 1].imshow(hist_s)
        axs[1, 2].set_title("GT vs SelfC 'v-Channel'")
        axs[1, 2].imshow(hist_v)

        # plt.savefig(output_path + '/' + seq.split("/")[-1][:-6] + '_hsv_diff.png')
        plt.show()
        plt.close()

print(means_h265)