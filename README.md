# CoTrFuse

## 1.Download pre-trained swin transformer model (Swin-T)
   * [Get pre-trained model in this link]
      (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/" and create dir 'chechpoint','test_log' in the root path.
## 2.Prepare data
   * You can also go to https://challenge.isic-archive.com/data/#2017 to acquire the ISIC2017 dataset. Process the label from the csv file for training. Change the imgs_train_path, imgs_val_path, imgs_test_path in the train_class_after_segmentation to the path of the corresponding path.
   * You can also go to https://www.kaggle.com/datasets/cf77495622971312010dd5934ee91f07ccbcfdea8e2f7778977ea8485c1914df to acquire the COVID-QU-Ex dataset.
## 3. Environment
   * Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.
## 4. Train/Test
   * Run the train script on the ISIC-2017 and the COVID-QU-Ex dataset. The batch size we used is 8 and 16. If you do not have enough GPU memory, the bacth size can be reduced to 6 or 12 to save memory. For more information, contact 904363330@qq.com.
   
   * Train
   
   ```
   python train_ISIC.py
   python train_COV.py
   ```
   
   * Test
   
   ```
   python test_ISIC.py
   python test_COV.py
   ```
 
## References
   * [TransUnet](https://github.com/Beckschen/TransUNet)
   * [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
   * [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
