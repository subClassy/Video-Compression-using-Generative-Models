import glob
import os
import shutil

src_path = '/media/himank/SSD/ss_dataset_bbox_r4/img_dir/train'
dest_path = '/media/himank/SSD/selfc_data/reorg_UVG/UVG_src/selfc_sdn_train'

os.makedirs(dest_path, exist_ok=True)
image_files = glob.glob(src_path + "/*.png")

train_list = []

for seq_no, seq in enumerate(image_files):
    # folder_name = seq.split("_")[-6:-3]
    # folder_name = '_'.join(folder_name)
    seq_folder_name = seq.split("/")[-1][:-4]
    # seq_folder_name = folder_name + '_' + str(seq_no).zfill(3)
    train_list.append(seq_folder_name)
    
    cur_seq_output_dir = os.path.join(dest_path, seq_folder_name)
    os.makedirs(cur_seq_output_dir, exist_ok=True)

    for i in range(3):
        # image_file = glob.glob(seq_path + f"/im{j}.jpg")[0]
        shutil.copy(seq, os.path.join(cur_seq_output_dir, f'im{i+1}.png'))

    # print(len(image_files))

with open(os.path.join(dest_path, 'test_list.txt'), 'w') as f:
    for row in train_list:
        f.write("%s\n" % str(row))