import glob
import os
import shutil

src_path = '/media/himank/SSD/selfc_data/reorg_UVG/UVG_src/selfc_mm'
dest_path = '/media/himank/SSD/selfc_data/mm_septuplet'

seq_output_dir = os.path.join(dest_path, 'sequences')
os.makedirs(seq_output_dir, exist_ok=True)

sequences = [d for d in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, d))]
gap = 3

train_list = []

for seq_no, seq in enumerate(sequences):
    seq_folder_name = str(seq_no+1).zfill(5)
    cur_seq_output_dir = os.path.join(seq_output_dir, seq_folder_name)
    os.makedirs(cur_seq_output_dir, exist_ok=True)

    seq_path = os.path.join(src_path, seq)
    image_files = glob.glob(seq_path + "/*.jpg")
    seq_len = len(image_files)
    for i in range(seq_len // 7):
        group_folder_name = str(i+1).zfill(4)
        cur_group_output_dir = os.path.join(cur_seq_output_dir, group_folder_name)
        os.makedirs(cur_group_output_dir, exist_ok=True)

        train_list.append(seq_folder_name + "/" + group_folder_name)

        count = 0
        for j in range((gap*i)+1, (gap*i)+8):
            count += 1
            image_file = glob.glob(seq_path + f"/im{j}.jpg")[0]
            shutil.copy(image_file, os.path.join(cur_group_output_dir, f'im{count}.jpg'))

    # print(len(image_files))

with open(os.path.join(dest_path, 'train_list.txt'), 'w') as f:
    for row in train_list:
        f.write("%s\n" % str(row))