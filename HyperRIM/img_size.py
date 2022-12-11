import glob
import cv2
import os
import shutil

src_path = '/media/himank/SSD/SelfC/results/test_codec_uvg_bf/selfc_hyperrim_train'
dest_path = '/media/himank/SSD/hyperrim_data/lr_images'

image_files = glob.glob(src_path + "/*.png")
count = 0

for seq_no, seq in enumerate(image_files):
    img_gt = cv2.imread(seq)
    if (img_gt.shape[0] == 1536//2 and img_gt.shape[1] == 2048//2):
        shutil.copy(seq, os.path.join(dest_path, seq.split('/')[-1]))
        count += 1

print(count)
