import numpy as np 
import pandas as pd 
import os
import SimpleITK as sitk


class ImgProcessing():
    def __init__(self,objimg):
        self.img = sitk.ReadImage(objimg)
        self.img_array = sitk.GetArrayFromImage(self.img)           #单通道
        # self.img_array = sitk.GetArrayFromImage(self.img)[:,:,0]   #三通道

    def inten_normal(self,obj_array=None,max_per=0.001,min_per=0.001):
        
        '''
        intensity normalization
        灰度归一化
        '''

        #   对象数组
        if obj_array:
            img_array = obj_array
        else:
            img_array = self.img_array
            #print(img_array.shape)
            #print(img_array.shape)
            #tmp_img = sitk.GetImageFromArray(img_array)
            #sitk.Show(tmp_img)
            #exit()
        N_I = img_array.astype(np.float32)
        I_sort = np.sort(N_I.flatten())
        I_min = I_sort[int(min_per*len(I_sort))]
        I_max = I_sort[-int(max_per*len(I_sort))]
        N_I =1.0*(N_I-I_min)/(I_max-I_min)
        N_I[N_I>1.0]=1.0  # 使大于1.0的数变为1.0
        N_I[N_I<0.0]=0.0
        self.img_array = N_I

    def z_score(self,obj_array=None):

        '''
        Z-score standardization
        零-均值规范化（z-score标准化）
        '''
        
        if obj_array:
            img_array = obj_array
        else:
            img_array = self.img_array
        img = (img_array-np.mean(img_array))/np.std(img_array)
        self.img_array = img

    def show(self):
        sitk.Show(self.img)
    
    def get_array(self):
        return self.img_array