import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import csv
import torch
from pathlib import Path
import shutil



SOURCE_ROOT = "/media/himank/SSD/selfc_mm_data"
TARGET_ROOT = "/media/himank/SSD/selfc_data/reorg_UVG/UVG_src/selfc_mm"

Path(TARGET_ROOT).mkdir(parents=True, exist_ok=True)

changes_idx = []
changes = []
img_list = sorted(os.listdir(SOURCE_ROOT))
time_diffs = []
test_list = []

curr_truck_path = os.path.join(TARGET_ROOT, img_list[0][:-4])
test_list.append(img_list[0][:-4])

Path(curr_truck_path).mkdir(parents=True, exist_ok=True)
count = 0

for curr in range(len(img_list) - 1):
    curr_time = int(img_list[curr].split("_")[2])
    next_time = int(img_list[curr+1].split("_")[2])
    time_diff = next_time - curr_time
    time_diffs.append(time_diff)
    count = count + 1
    shutil.copy(os.path.join(SOURCE_ROOT, img_list[curr]), os.path.join(TARGET_ROOT, curr_truck_path, f'im{count}.jpg'))
    if time_diff >= 30:
        count = 0
        curr_truck_path = os.path.join(TARGET_ROOT, img_list[curr + 1][:-4])
        test_list.append(img_list[curr + 1][:-4])
        Path(curr_truck_path).mkdir(parents=True, exist_ok=True)
    # if curr >= 50:
    #     exit(0)
shutil.copy(os.path.join(SOURCE_ROOT, img_list[-1]), os.path.join(TARGET_ROOT, curr_truck_path, f'im{count}.jpg'))

with open(os.path.join(TARGET_ROOT, 'test_list.txt'), 'w') as f:
    for row in test_list:
        f.write("%s\n" % str(row))


# for i in range(len(time_diffs)):
#     if time_diffs[i] >= 30:
#         changes_idx.append(i)
#         changes.append(time_diffs[i])
# plt.plot([i for i in range(len(time_diffs))], time_diffs)
# plt.scatter(changes_idx, changes, color="red")
# plt.title =  "Time Diff"
# plt.xlabel = "First Image Index"
# plt.ylabel = "Time Difference"
# plt.show()
# plt.savefig("test")

# with open('time_diff', 'w') as f:
      
#     # using csv.writer method from CSV package
#     writer = csv.writer(f)
#     writer.writerow(time_diffs)
#     print("Done")
