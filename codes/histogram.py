import glob
import cv2
import matplotlib.pyplot as plt

src_path = '/media/himank/SSD/SelfC/results/test_codec_uvg_bf/selfc_mm_rocks_big'
# src_path = '/media/himank/SSD/histo_matched/histo_matched_rgb'

image_files = glob.glob(src_path + "/*gt.jpg")

def get_histogram(img):
    ch1, ch2, ch3 = img[:,:,0], img[:,:,1], img[:,:,2]
    hist_b = cv2.calcHist([ch1],[0],None,[256],[0,256])
    hist_g = cv2.calcHist([ch2],[0],None,[256],[0,256])
    hist_r = cv2.calcHist([ch3],[0],None,[256],[0,256])

    return hist_b, hist_g, hist_r

for seq_no, seq in enumerate(image_files):
    fig, axs = plt.subplots(2, 3, figsize=(24, 12))
    
    img_gt = cv2.imread(seq)
    img_h265 = cv2.imread(seq[:-6] + 'h265.jpg')
    img_selfc = cv2.imread(seq[:-6] + 'selfc.jpg')
    
    hist_b, hist_g, hist_r = get_histogram(img_gt)
    axs[0, 0].plot(hist_r, color='r', label="r")
    axs[0, 0].plot(hist_g, color='g', label="g")
    axs[0, 0].plot(hist_b, color='b', label="b")
    axs[0, 0].set_title("GT")
    axs[0, 0].set_ylim(0, 150000)
    axs[0, 0].legend()
    
    hist_b, hist_g, hist_r = get_histogram(img_h265)
    axs[0, 1].plot(hist_r, color='r', label="r")
    axs[0, 1].plot(hist_g, color='g', label="g")
    axs[0, 1].plot(hist_b, color='b', label="b")
    axs[0, 1].set_title("H265")
    axs[0, 1].set_ylim(0, 150000)
    axs[0, 1].legend()
    
    hist_b, hist_g, hist_r = get_histogram(img_selfc)
    axs[0, 2].plot(hist_r, color='r', label="r")
    axs[0, 2].plot(hist_g, color='g', label="g")
    axs[0, 2].plot(hist_b, color='b', label="b")
    axs[0, 2].set_title("SelfC")
    axs[0, 2].set_ylim(0, 150000)
    axs[0, 2].legend()
    
    
    img2_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2HSV)
    img2_h265 = cv2.cvtColor(img_h265, cv2.COLOR_BGR2HSV)
    img2_selfc = cv2.cvtColor(img_selfc, cv2.COLOR_BGR2HSV)

    hist_h, hist_s, hist_v = get_histogram(img2_gt)
    axs[1, 0].plot(hist_h, color='c', label="h")
    axs[1, 0].plot(hist_s, color='m', label="s")
    axs[1, 0].plot(hist_v, color='y', label="v")
    axs[1, 0].legend()
    axs[1, 0].set_ylim(0, 600000)

    hist_h, hist_s, hist_v = get_histogram(img2_h265)
    axs[1, 1].plot(hist_h, color='c', label="h")
    axs[1, 1].plot(hist_s, color='m', label="s")
    axs[1, 1].plot(hist_v, color='y', label="v")
    axs[1, 1].legend()
    axs[1, 1].set_ylim(0, 600000)
    
    hist_h, hist_s, hist_v = get_histogram(img2_selfc)
    axs[1, 2].plot(hist_h, color='c', label="h")
    axs[1, 2].plot(hist_s, color='m', label="s")
    axs[1, 2].plot(hist_v, color='y', label="v")
    axs[1, 2].legend()
    axs[1, 2].set_ylim(0, 600000)
    
    plt.savefig(seq[:-6] + 'hist.png')
    # plt.show()
    plt.close()
