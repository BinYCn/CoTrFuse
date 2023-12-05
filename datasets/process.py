import pandas as pd
import cv2
import os


image_path = ''  # path for image
mask_path = ''  # path for mask
csv_path = ''  # path for Data_metadata.csv
save_path = ''  # path for save_path


save_image_path = os.path.join(save_path, 'image')
save_mask_path = os.path.join(save_path, 'mask')

if not os.path.exists(save_image_path):
    os.mkdir(save_image_path)

if not os.path.exists(save_mask_path):
    os.mkdir(save_image_path)


for name in pd.read_csv(csv_path)['image_id'].to_list():
    image = os.path.join(image_path, name + '.jpg')
    mask = os.path.join(mask_path, name + '_segmentation.png')
    resize_image = cv2.resize(cv2.imread(image), (512, 512))
    resize_mask = cv2.resize(cv2.imread(image), (512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(save_image_path, name + '.jpg'), resize_image)
    cv2.imwrite(os.path.join(save_mask_path, name + '_segmentation.png'), resize_mask)

df = pd.DataFrame()
df['image_id'] = pd.read_csv(csv_path)['image_id'].to_list()
df.to_csv(os.path.join(save_path, 'image_id.csv'), index=False)
