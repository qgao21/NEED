import os
import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import numpy as np
import torch
from functools import partial
import torch.nn.functional as F
import math
import random
import cv2
import sys
import copy
from skimage import transform

from ipdb import set_trace


class CTDataset(Dataset):
    def __init__(self, npy_root, mode, context=True, data_type='img', norm_min=-1024, norm_max=3072, model_name=None):
        self.mode = mode
        self.context = context
        self.data_type = data_type
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.model_name = model_name

                   # 560, 823, 318, 585, 600, 525, 856, 533, 610, 526
        # patient_ids = [67, 96, 109, 143, 192, 286, 291, 310, 333, 506]
        if mode == 'train':
            # patient_ids = patient_ids[:test_id]
            patient_ids = [67, 96, 109, 192, 286, 291, 310, 333]
        elif mode == 'test':
            # patient_ids = patient_ids[test_id:]
            patient_ids = [143, 506]
        elif mode == 'mayo2020':
            # patient_ids = ['C052', 'C232', 'C016', 'C120', 'C050', 'L077', 'L056', 'L186', 'L006', 'L148']
            patient_ids = ['C052', 'C232', 'C016', 'C120', 'C050', 'L077', 'L056', 'L186', 'L006', 'L148']
            if data_type == 'img':
                data_root = osp.join(npy_root, 'Dataset/gen_data/mayo_2020_sim_img_npy/')
            elif data_type == 'sino':
                data_root = osp.join(npy_root, 'Dataset/gen_data/mayo_2020_sim_sino_npy/')

            middle_slices = np.array([169, 512, 844, 1173, 1516, 1766, 1889, 2016, 2169, 2317])
            for i in range(10):
                middle_slice = middle_slices[i]
                if i == 0:
                    select_slices = np.arange(middle_slice - 40, middle_slice + 40)
                else:
                    select_slices = np.append(select_slices, np.arange(middle_slice - 40, middle_slice + 40))
            # print(select_slices)
            # set_trace()


        if mode in ['train', 'test']:
            if data_type == 'img':
                # data_root = osp.join(npy_root, 'Dataset/gen_data/mayo_2016_astra_sim_torch_radon_img/')
                if model_name.startswith('sr'):
                    data_root = osp.join(npy_root, 'Dataset/gen_data/mayo_2016_astra_sim_torch_radon_img2/')
                    lr_data_root = osp.join(npy_root, 'Dataset/gen_data/mayo_2016_astra_sim_torch_radon_img2_256/')
                else:
                    data_root = osp.join(npy_root, 'Dataset/gen_data/mayo_2016_astra_sim_torch_radon_img2/')
                n2n_data_root = osp.join(npy_root, 'Dataset/gen_data/mayo_2016_astra_sim_torch_radon_img2_n2n/')
            elif data_type == 'sino':
                # data_root = osp.join(npy_root, 'Dataset/gen_data/mayo_2016_astra_sim_torch_radon_sino/')
                data_root = osp.join(npy_root, 'Dataset/gen_data/mayo_2016_astra_sim_torch_radon_sino2/')
                n2n_data_root = osp.join(npy_root, 'Dataset/gen_data/mayo_2016_astra_sim_torch_radon_sino2_n2n/')

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('L{:03d}_'.format(id) + '*_0_' + data_type +'.npy'))))
                if context and model_name.startswith('sr') == False and model_name!='corediff': #and mode=='train':
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list)-1]
                    patient_lists = patient_lists + patient_list
            base_target = patient_lists



            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('L{:03d}_'.format(id) + '*_{}_'.format(dose) + data_type + '.npy'))))
                if context:
                    cat_patient_list = []
                    for i in range(1, len(patient_list)-1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i+j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
            base_input = patient_lists

        elif mode == 'mayo2020':
            patient_lists = []
            for ind, id in enumerate(patient_ids):
                if self.data_type == 'img':
                    patient_list = sorted(glob(osp.join(data_root, id + '_' + '*_target_radon_{}.npy'.format(data_type))))
                else:
                    patient_list = sorted(glob(osp.join(data_root, id + '_' + '*_target_{}.npy'.format(data_type))))
                if context:# and model_name!='corediff':
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
            base_target = patient_lists
            base_target = [base_target[i] for i in select_slices]

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                if self.data_type == 'img':
                    patient_list = sorted(glob(osp.join(data_root, id + '_' + '*_input_radon_{}.npy'.format(data_type))))
                else:
                    patient_list = sorted(glob(osp.join(data_root, id + '_' + '*_input_{}.npy'.format(data_type))))

                if context:
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
            base_input = patient_lists
            base_input = [base_input[i] for i in select_slices]

        self.input = base_input
        self.target = base_target
        print(len(self.input))
        print(len(self.target))


    def __getitem__(self, index):
        input, target = self.input[index], self.target[index]

        if self.context:
            input = input.split('~')
            inputs = []
            for i in range(1, len(input)):
                inputs.append(self.norm_(np.load(input[i])[np.newaxis, ...].astype(np.float32), self.norm_min, self.norm_max))
            input = np.concatenate(inputs, axis=0)  # [3, 512, 512]
        else:
            if self.model_name == 'pdf_redcnn':
                input = input.split('~')
                inputs = []
                for i in range(0, len(input)):
                    inputs.append(self.norm_(np.load(input[i])[np.newaxis, ...].astype(np.float32), self.norm_min, self.norm_max))
                input = np.concatenate(inputs, axis=0)  # [4, 512, 512]
            else:
                input = self.norm_(np.load(input)[np.newaxis, ...].astype(np.float32), self.norm_min, self.norm_max)  # [1, 512, 512]
        # if sino, size = [1, angles, detect_num]
        if self.model_name.startswith('sr'):
            c, h, w = input.shape
            input = transform.resize(input, (c, 512, 512), order=3)

        if self.context and self.model_name.startswith('sr') == False and (self.model_name!='corediff' or self.mode in ['pulmonary', 'mayo2020']): #and self.mode=='train':
            target = target.split('~')
            targets = []
            for i in range(1, len(target)):
                targets.append(self.norm_(np.load(target[i])[np.newaxis, ...].astype(np.float32), self.norm_min, self.norm_max))
            target = np.concatenate(targets, axis=0)  # [3, 512, 512]
        else:
            target = self.norm_(np.load(target)[np.newaxis, ...].astype(np.float32), self.norm_min, self.norm_max)  # [1, 512, 512]

        # if self.data_type == 'img':
        #     c, h, w = input.shape
        #     input = transform.resize(input, (c, h//2, w//2))
        #     c, h, w = target.shape
        #     target = transform.resize(target, (c, h // 2, w // 2))
        # cv2.imwrite('input.png', 255 * input.transpose(1, 2, 0))


    def __len__(self):
        return len(self.target)


    def norm_(self, img, MIN_B=-1024, MAX_B=3072):
        if self.data_type == 'img':
            img = img - 1024
            img[img < MIN_B] = MIN_B
            img[img > MAX_B] = MAX_B
            img = (img - MIN_B) / (MAX_B - MIN_B)
        elif self.data_type == 'sino':
            img = img / 1000
        return img


dataset_dict = {
    'train': partial(CTDataset, npy_root='', mode='train', dose=2, context=True, data_type='img', norm_min=-1024, norm_max=3072),
    'test': partial(CTDataset, npy_root='', mode='test', dose=2, context=True, data_type='img', norm_min=-1024, norm_max=3072),
    'mayo2020': partial(CTDataset, npy_root='', mode='mayo2020', dose=2, context=True, data_type='img', norm_min=-1024, norm_max=3072)
}