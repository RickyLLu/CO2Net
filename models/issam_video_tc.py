from functools import partial

import torch
from torchvision import transforms
import time
import os
import numpy as np
from easydict import EasyDict as edict
from albumentations import HorizontalFlip, Resize, RandomResizedCrop

from iharm.data.compose import ComposeDataset, MyPreviousDataset, MyPreviousSequenceDataset
from iharm.data.hdataset import HDataset
from iharm.data.transforms import HCompose
from iharm.engine.simple_trainer import SimpleHTrainer, SimpleHTrainer_tc_with_lutoutput
from iharm.model import initializer
from iharm.model.base import SSAMvideoLut, SSAMvideoLutTC
from iharm.model.losses import MaskWeightedMSE, MaskWeightedMSE_tc
from iharm.model.metrics import DenormalizedMSEMetric, DenormalizedPSNRMetric, MSEMetric
from iharm.utils.log import logger


class my_total_dataset:
    def __init__(self, val_list, dataset_path, input_transform=None, augmentator=None, previous_num = 0, future_num = 0,
                 keep_background_prob = -1):
        start_time = time.time()
        self.tasks = []
        self.dataset_path = dataset_path
        self.input_transform = input_transform
        self.augmentator = augmentator
        with open(val_list, 'r') as f:
            for line in f.readlines():
                tar_name, mask_name, cur_name = line.split()
                tar_name = tar_name.replace('\\', '/')
                mask_name = mask_name.replace('\\', '/')
                cur_name = cur_name.replace('\\', '/')
                self.tasks.append([tar_name, mask_name, cur_name])
        self.previous_num = previous_num
        self.future_num = future_num
        self.trans_img_names = []
        self.trans_img_dic = {}
        self.masks_names = []
        self.mask_dic = {}
        self.keep_background_prob = keep_background_prob
        self.target_img_names = []
        self.new_tasks = []
        self.tar_img_dic = {}
        self.trans_imgs = []
        self.masks = []
        self.tar_imgs = []
        for i in range(len(self.tasks)):
            if (i%100==0):
                print(i)
            tar_name, mask_name, cur_name = self.tasks[i]
            video_name, obj_name, number = cur_name.split('/')[-3:]

            #tar_img_name = os.path.join(self.dataset_path, tar_name)
            #mask_img_name = os.path.join(self.dataset_path, mask_name)
            #cur_img_name = os.path.join(self.dataset_path, cur_name)

            cur_img_name = '/home/ubuntu/256_dataset/trans_img/' + video_name + '_' + obj_name + '_' + number[:-4] + '.npy'
            mask_img_name = '/home/ubuntu/256_dataset/mask/' + video_name + '_' + obj_name + '_' + number[:-4] + '.npy'
            tar_img_name = '/home/ubuntu/256_dataset/target_img/' + video_name + '_' + number[:-4] + '.npy'
            assert os.path.exists(tar_img_name)
            assert os.path.exists(mask_img_name)
            assert os.path.exists(cur_img_name)
            self.trans_img_dic[cur_name] = i
            self.mask_dic[mask_name] = i
            if self.tasks[i][0] not in self.tar_img_dic:
                self.tar_img_dic[self.tasks[i][0]] = len(self.tar_imgs)
                #target_image = cv2.imread(tar_img_name)
                target_image = np.load(tar_img_name)
                self.tar_imgs.append(target_image)
            #mask_img = cv2.imread(mask_img_name)
            #mask_img = mask_img[:, :, 0].astype(np.float32) / 255.
            mask_img = np.load(mask_img_name)
            self.masks.append(mask_img)
            #trans_img = cv2.imread(cur_img_name)
            trans_img = np.load(cur_img_name)
            self.trans_imgs.append(trans_img)
            self.new_tasks.append((self.tar_img_dic[self.tasks[i][0]], i, i))
        end_time = time.time()
        print("initialize cost:", end_time - start_time)

    def __getitem__(self, index):
        sample = {}
        tar_name, mask_name, cur_name = self.tasks[index]

        cur_img = self.trans_imgs[self.trans_img_dic[cur_name]]
        cur_mask = self.masks[self.mask_dic[mask_name]]
        tar_img = self.tar_imgs[self.tar_img_dic[tar_name]]




        pre_imgs = []
        pre_masks = []
        future_imgs = []
        future_masks = []
        #print(os.path.split(cur_name))
        #print(cur_name)
        path, number = os.path.split(cur_name)
        mask_path, mask_number = os.path.split(mask_name)
        for p in range(1, self.previous_num+1):
            pre_number = '%05d' % (int(number[:-4]) - 5 * p) + number[-4:]
            pre_mask_number = '%05d' % (int(mask_number[:-4]) - 5 * p) + mask_number[-4:]
            pre_img_name = os.path.join(path, pre_number)
            pre_mask_name = os.path.join(mask_path, pre_mask_number)
            if pre_img_name in self.trans_img_dic:
                pre_imgs.append(self.trans_imgs[self.trans_img_dic[pre_img_name]])
                pre_masks.append(self.masks[self.mask_dic[pre_mask_name]])
            else:
                if len(pre_imgs) > 0:
                    tmp_img = pre_imgs[-1]
                    pre_imgs.append(tmp_img)
                    tmp_mask = pre_masks[-1]
                    pre_masks.append(tmp_mask)
                else:
                    pre_imgs.append(cur_img)
                    pre_masks.append(cur_mask)

        for p in range(1, self.future_num+1):
            future_number = '%05d' % (int(number[:-4]) + 5 * p) + number[-4:]
            future_mask_number = '%05d' % (int(mask_number[:-4]) + 5 * p) + mask_number[-4:]
            future_img_name = os.path.join(path, future_number)
            future_mask_name = os.path.join(mask_path, future_mask_number)
            if future_img_name in self.trans_img_dic:
                future_imgs.append(self.trans_imgs[self.trans_img_dic[future_img_name]])
                future_masks.append(self.masks[self.mask_dic[future_mask_name]])
            else:
                if len(future_imgs) > 0:
                    tmp_img = future_imgs[-1]
                    future_imgs.append(tmp_img)
                    tmp_mask = future_masks[-1]
                    future_masks.append(tmp_mask)
                else:
                    future_imgs.append(cur_img)
                    future_masks.append(cur_mask)

        pre_imgs += future_imgs
        pre_masks += future_masks
        '''
        new_pre_imgs = []
        new_pre_masks = []
        for pp in range(len(pre_imgs)):
            tmp = self.augment_sample({
            'image': cur_img,
            'object_mask': cur_mask,
            'target_image': tar_img,
            'image_id': index,
            'pre_image': pre_imgs[pp],
            'pre_object_mask': pre_masks[pp],
        })
            new_pre_imgs.append(tmp['pre_image'])
            new_pre_masks.append(tmp['pre_object_mask'])
            new_img = tmp['image']
            new_object_mask = tmp['object_mask']
            new_target_img = tmp['target_image']

        '''
        new_img = cur_img
        new_target_img = tar_img
        new_object_mask = cur_mask
        new_pre_imgs = pre_imgs
        new_pre_masks = pre_masks

        cur_img = self.input_transform(new_img)
        #cur_mask = self.input_transform(cur_mask)
        tar_img = self.input_transform(new_target_img)

        sample['images'] = cur_img
        sample['masks'] = new_object_mask[np.newaxis, ...].astype(np.float32)
        sample['target_images'] = tar_img

        for j in range(len(new_pre_imgs)):
            new_pre_imgs[j] = self.input_transform(new_pre_imgs[j])
            new_pre_masks[j] = new_pre_masks[j].astype(np.float32)
            new_pre_masks[j] = new_pre_masks[j][np.newaxis, ...].astype(np.float32)

        new_pre_imgs = torch.stack(new_pre_imgs, dim=0)
        new_pre_masks = np.array(new_pre_masks)

        sample['pre_images'] = new_pre_imgs
        sample['pre_masks'] = new_pre_masks
        sample['name'] = cur_name

        return sample

    def __len__(self):
        return len(self.tasks)


    def augment_sample(self, sample):
        if self.augmentator is None:
            return sample

        additional_targets = {target_name: sample[target_name]
                              for target_name in self.augmentator.additional_targets.keys()}

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.augmentator(image=sample['image'], **additional_targets)
            valid_augmentation = self.check_augmented_sample(sample, aug_output)

        for target_name, transformed_target in aug_output.items():
            sample[target_name] = transformed_target

        return sample

    def check_sample_types(self, sample):
        assert sample['image'].dtype == 'uint8'
        if 'target_image' in sample:
            assert sample['target_image'].dtype == 'uint8'

    def check_augmented_sample(self, sample, aug_output):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True

        return aug_output['object_mask'].sum() > 1.0

class my_total_dataset_with_lut_result:
    def __init__(self, val_list, dataset_path, input_transform=None,  previous_num=0, future_num=0,
                 keep_background_prob=-1):
        start_time = time.time()
        self.tasks = []
        self.dataset_path = dataset_path
        self.input_transform = input_transform
        with open(val_list, 'r') as f:
            for line in f.readlines():
                tar_name, mask_name, cur_name = line.split()
                tar_name = tar_name.replace('\\', '/')
                mask_name = mask_name.replace('\\', '/')
                cur_name = cur_name.replace('\\', '/')
                self.tasks.append([tar_name, mask_name, cur_name])
        self.previous_num = previous_num
        self.future_num = future_num
        self.trans_img_names = []
        self.trans_img_dic = {}
        self.masks_names = []
        self.mask_dic = {}
        self.keep_background_prob = keep_background_prob
        self.target_img_names = []
        self.new_tasks = []
        self.tar_img_dic = {}
        self.trans_imgs = []
        self.masks = []
        self.tar_imgs = []
        for i in range(len(self.tasks)):
            if (i % 100 == 0):
                print(i)
            tar_name, mask_name, cur_name = self.tasks[i]
            video_name, obj_name, number = cur_name.split('/')[-3:]

            # tar_img_name = os.path.join(self.dataset_path, tar_name)
            # mask_img_name = os.path.join(self.dataset_path, mask_name)
            # cur_img_name = os.path.join(self.dataset_path, cur_name)

            cur_img_name = '/home/ubuntu/256_dataset/trans_img/' + video_name + '_' + obj_name + '_' + number[
                                                                                                       :-4] + '.npy'
            mask_img_name = '/home/ubuntu/256_dataset/mask/' + video_name + '_' + obj_name + '_' + number[:-4] + '.npy'
            tar_img_name = '/home/ubuntu/256_dataset/target_img/' + video_name + '_' + number[:-4] + '.npy'
            assert os.path.exists(tar_img_name)
            assert os.path.exists(mask_img_name)
            assert os.path.exists(cur_img_name)
            self.trans_img_dic[cur_name] = i
            self.mask_dic[mask_name] = i
            if self.tasks[i][0] not in self.tar_img_dic:
                self.tar_img_dic[self.tasks[i][0]] = len(self.tar_imgs)
                # target_image = cv2.imread(tar_img_name)
                target_image = np.load(tar_img_name)
                self.tar_imgs.append(target_image)
            # mask_img = cv2.imread(mask_img_name)
            # mask_img = mask_img[:, :, 0].astype(np.float32) / 255.
            mask_img = np.load(mask_img_name)
            self.masks.append(mask_img)
            # trans_img = cv2.imread(cur_img_name)
            trans_img = np.load(cur_img_name)
            self.trans_imgs.append(trans_img)
            self.new_tasks.append((self.tar_img_dic[self.tasks[i][0]], i, i))
        end_time = time.time()
        print("initialize cost:", end_time - start_time)

    def __getitem__(self, index):
        sample = {}
        tar_name, mask_name, cur_name = self.tasks[index]

        cur_img = self.trans_imgs[self.trans_img_dic[cur_name]]
        cur_mask = self.masks[self.mask_dic[mask_name]]
        tar_img = self.tar_imgs[self.tar_img_dic[tar_name]]
        video, obj, img_number = cur_name.split('/')[-3:]
        new_name = '/lut_output/20_20/' + video + '_' + obj + '_' + img_number[:-3] + 'npy'

        new_img = cur_img
        new_target_img = tar_img
        new_object_mask = cur_mask

        cur_img = self.input_transform(new_img)
        # cur_mask = self.input_transform(cur_mask)
        tar_img = self.input_transform(new_target_img)
        lut_output = np.load(new_name)
        lut_output = torch.from_numpy(lut_output)
        sample['images'] = cur_img
        sample['masks'] = new_object_mask[np.newaxis, ...].astype(np.float32)
        sample['target_images'] = tar_img
        sample['name'] = cur_name
        sample['lut_output'] = lut_output

        return sample

    def __len__(self):
        return len(self.tasks)

    def augment_sample(self, sample):
        if self.augmentator is None:
            return sample

        additional_targets = {target_name: sample[target_name]
                              for target_name in self.augmentator.additional_targets.keys()}

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.augmentator(image=sample['image'], **additional_targets)
            valid_augmentation = self.check_augmented_sample(sample, aug_output)

        for target_name, transformed_target in aug_output.items():
            sample[target_name] = transformed_target

        return sample

    def check_sample_types(self, sample):
        assert sample['image'].dtype == 'uint8'
        if 'target_image' in sample:
            assert sample['target_image'].dtype == 'uint8'

    def check_augmented_sample(self, sample, aug_output):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True

        return aug_output['object_mask'].sum() > 1.0



def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg, start_epoch=cfg.start_epoch)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (256, 256)
    model_cfg.input_normalization = {
        'mean': [.485, .456, .406],
        'std': [.229, .224, .225]
    }
    model_cfg.depth = 4

    model_cfg.input_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(model_cfg.input_normalization['mean'], model_cfg.input_normalization['std']),
    ])

    model = SSAMvideoLutTC(
        depth=4, ssam_backbone = cfg.ssam_backbone, with_lutoutput = False
    )
    #model.init_device(cfg.device)
    model.to(cfg.device)
    #print(model.state_dict()['issam.decoder.up_blocks.0.upconv.1.block.0.weight'].sum())
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=1.0))
    model.load_backbone()
    #print(model.state_dict()['issam.decoder.up_blocks.0.upconv.1.block.0.weight'].sum())
    return model, model_cfg


def train(model, cfg, model_cfg, start_epoch=0):
    cfg.batch_size = 16 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size

    cfg.input_normalization = model_cfg.input_normalization
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.pixel_loss = MaskWeightedMSE()
    loss_cfg.pixel_loss_weight = 1.0

    loss_cfg.tc_pixel_loss = MaskWeightedMSE_tc()
    loss_cfg.tc_pixel_loss_weight = 1.0

    num_epochs = cfg.total_epoch

    train_augmentator = HCompose([
        Resize(*crop_size),
        HorizontalFlip(),
    ])

    val_augmentator = HCompose([
        Resize(*crop_size)
    ])
    previous_number = cfg.previous_num
    future_number = cfg.future_num
    '''

    trainset = my_total_dataset_with_lut_result(cfg.train_list, cfg.dataset_path,previous_num =previous_number, future_num=future_number,
        input_transform=model_cfg.input_transform)
    valset = my_total_dataset_with_lut_result(cfg.val_list, cfg.dataset_path, previous_num =previous_number, future_num=future_number,
        input_transform=model_cfg.input_transform)
    '''
    trainset = my_total_dataset(cfg.train_list, cfg.dataset_path, previous_num=previous_number,
                                                future_num=future_number,
                                                input_transform=model_cfg.input_transform)
    valset = my_total_dataset(cfg.val_list, cfg.dataset_path, previous_num=previous_number,
                                              future_num=future_number,
                                              input_transform=model_cfg.input_transform)

    optimizer_params = {
        'lr': 1e-3,
        'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[105, 115], gamma=0.1)
    trainer = SimpleHTrainer_tc_with_lutoutput(
        model, cfg, model_cfg, loss_cfg,
        trainset, valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        metrics=[

            MSEMetric('images', 'target_images')
        ],
        checkpoint_interval=10,
        image_dump_interval=1000,
        with_previous=True
    )


    logger.info(f'Starting Epoch: {start_epoch}')
    logger.info(f'Total Epochs: {num_epochs}')

    for epoch in range(start_epoch, num_epochs):
        trainer.training(epoch)
        trainer.validation(epoch)

