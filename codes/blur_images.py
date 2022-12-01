import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

src_path = '/media/himank/SSD/SelfC/results/test_codec_uvg_bf/selfc_mm_rocks_big'
output_path = '/media/himank/SSD/SelfC/results/test_codec_uvg_bf/selfc_mm_rocks_big/blurrred_images'

os.makedirs(output_path, exist_ok=True)

# src_path = '/media/himank/SSD/histo_matched/histo_matched_rgb'

image_files = glob.glob(src_path + "/*selfc.jpg")

def get_difference(img_gt, img_recon):
    ch1, ch2, ch3 = img_gt[:,:,0], img_gt[:,:,1], img_gt[:,:,2]
    ch1_r, ch2_r, ch3_r = img_recon[:,:,0], img_recon[:,:,1], img_recon[:,:,2]

    hist_b = np.abs(np.array(ch1, dtype=np.float32) - np.array(ch1_r, dtype=np.float32))
    hist_g = np.abs(np.array(ch2, dtype=np.float32) - np.array(ch2_r, dtype=np.float32))
    hist_r = np.abs(np.array(ch3, dtype=np.float32) - np.array(ch3_r, dtype=np.float32))

    return hist_b, hist_g, hist_r

draw_plot = False

for seq_no, seq in enumerate(image_files):
    img_selfc = cv2.imread(seq)
    
    img_blurred = cv2.GaussianBlur(img_selfc, [5, 5], 0)
    cv2.imwrite(os.path.join(output_path, seq.split('/')[-1]), img_blurred)
    
    if draw_plot:
        fig, axs = plt.subplots(1, 2, figsize=(24, 12))
    
        axs[0].set_title("Original Img")
        axs[0].imshow(img_selfc)
        axs[1].set_title("Blurred Img")
        axs[1].imshow(img_blurred)

        plt.show()
        plt.close()

    