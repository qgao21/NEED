import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import numpy as np
from functools import partial


class CTDataset(Dataset):
    def __init__(self, npy_root, mode, dose=2, context=True, data_type='img', norm_min=-1024, norm_max=3072):
        self.mode = mode
        self.dose = dose
        self.context = context
        self.data_type = data_type
        self.norm_min = norm_min
        self.norm_max = norm_max

        if mode == 'train':
            patient_ids = [67, 96, 109, 192, 286, 291, 310, 333]
        elif mode == 'test':
            patient_ids = [143, 506]
        elif mode == 'mayo2020':
            patient_ids = ['C052', 'C232', 'C016', 'C120', 'C050', 'L077', 'L056', 'L186', 'L006', 'L148']
            if data_type == 'img':
                data_root = osp.join(npy_root, 'mayo2020_img_path')
            elif data_type == 'sino':
                data_root = osp.join(npy_root, 'mayo2020_sino_path/')

            middle_slices = np.array([169, 512, 844, 1173, 1516, 1766, 1889, 2016, 2169, 2317])
            for i in range(10):
                middle_slice = middle_slices[i]
                if i == 0:
                    select_slices = np.arange(middle_slice - 40, middle_slice + 40)
                else:
                    select_slices = np.append(select_slices, np.arange(middle_slice - 40, middle_slice + 40))

        if mode in ['train', 'test']:
            if data_type == 'img':
                data_root = osp.join(npy_root, 'mayo2016_img_path/')
            elif data_type == 'sino':
                data_root = osp.join(npy_root, 'mayo2016_sino_path/')

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('L{:03d}_'.format(id) + '*_0_' + data_type +'.npy'))))
                if context:
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
                    patient_list = sorted(glob(osp.join(data_root, id + '_' + '*_target_{}.npy'.format(data_type))))
                else:
                    patient_list = sorted(glob(osp.join(data_root, id + '_' + '*_target_{}.npy'.format(data_type))))
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
            base_target = patient_lists
            base_target = [base_target[i] for i in select_slices]

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                if self.data_type == 'img':
                    patient_list = sorted(glob(osp.join(data_root, id + '_' + '*_input_{}.npy'.format(data_type))))
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
            input = self.norm_(np.load(input)[np.newaxis, ...].astype(np.float32), self.norm_min, self.norm_max)
            # if img, size = [1, 512, 512]
            # if sino, size = [1, angles, detect_num]

        if self.context:
            target = target.split('~')
            targets = []
            for i in range(1, len(target)):
                targets.append(self.norm_(np.load(target[i])[np.newaxis, ...].astype(np.float32), self.norm_min, self.norm_max))
            target = np.concatenate(targets, axis=0)  # [3, 512, 512]
        else:
            target = self.norm_(np.load(target)[np.newaxis, ...].astype(np.float32), self.norm_min, self.norm_max)

        return input, target

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

npy_root = 'your root path'
dataset_dict = {
    'train': partial(CTDataset, npy_root=npy_root, mode='train', dose=2, context=True, data_type='img', norm_min=-1024, norm_max=3072),
    'test': partial(CTDataset, npy_root=npy_root, mode='test', dose=2, context=True, data_type='img', norm_min=-1024, norm_max=3072),
    'mayo2020': partial(CTDataset, npy_root=npy_root, mode='mayo2020', dose=2, context=True, data_type='img', norm_min=-1024, norm_max=3072)
}
