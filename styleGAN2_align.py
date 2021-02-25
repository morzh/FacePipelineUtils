
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import os.path
import os
import json
from my_procrustes import *
from matplotlib import pyplot as plt
import shutil

path_src = '/media/morzh/ext4_volume/data/Faces/BeautifyMeFaceset-004/01_Neutral'
path_dst = '/media/morzh/ext4_volume/data/Faces/BeautifyMeFaceset-005/01_Neutral'
folders = ['']
# folders = ['Asians', 'Caucasians', 'asians_2add', 'caucasians2add']
# folders = ['asians_2add', 'caucasians2add']
os.makedirs(path_dst, exist_ok=True)
show_plot = False

avg_face_path = '/media/morzh/ext4_volume/data/Faces/ConvexHullFaceset-Neutral/aux'
avg_face_filename = 'stylegan2_avgface.png.lms103'

avg_lms = np.loadtxt(os.path.join( avg_face_path, avg_face_filename))
avg_face_landmarks = np.vstack((avg_lms[0:3], avg_lms[14:17], avg_lms[28:75], avg_lms[78:81], avg_lms[93:96]))

for folder in folders:
    pathfolder_src = os.path.join(path_src, folder)
    pathfolder_dst = os.path.join(path_dst, folder)
    os.makedirs(pathfolder_dst, exist_ok=True)
    files = [f for f in os.listdir(pathfolder_src) if f.endswith('.jpg') or f.endswith('png')]

    for file in files:
        pathfolderfile_src = os.path.join(pathfolder_src, file)

        img = cv2.imread(pathfolderfile_src)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        lms = np.loadtxt(pathfolderfile_src + '.lms103')

        landmarks = np.vstack((lms[0:3], lms[14:17], lms[28:75], lms[78:81], lms[93:96]))

        lms_aligned, lms_xform = my_procrustes(avg_face_landmarks, landmarks)
        lms_xformed = transform_lms(lms, np.transpose(lms_xform))
        M = lms_xform[0:2, :]
        img_aligned = cv2.warpAffine(img, M, (1024, 1024), flags=cv2.INTER_CUBIC)

        img_aligned_mask = img_aligned[:, :, 3]/255.0
        img_aligned_mask = np.dstack((img_aligned_mask, img_aligned_mask, img_aligned_mask))

        img_aligned = img_aligned[:, :, 0:3]
        img = img[:, :, 0:3]
        if img.shape[0:2] == (1024, 1024):
            img_merged = np.uint8(img_aligned*img_aligned_mask + img*(1-img_aligned_mask))
        else:
            img_merged = img_aligned

        if show_plot:
            fig, ax = plt.subplots()
            fig.set_size_inches(9, 10)
            plt.imshow(cv2.cvtColor(img_merged, cv2.COLOR_RGBA2BGRA))
            plt.scatter(avg_face_landmarks[:, 0], avg_face_landmarks[:, 1], c='r')
            plt.scatter(lms_xformed[:, 0], lms_xformed[:, 1], c='b', s=10)
            plt.tight_layout()
            plt.show()


        pathfolderfile_dst = os.path.join(pathfolder_dst, file)
        cv2.imwrite(pathfolderfile_dst, img_merged)
        np.savetxt(pathfolderfile_dst + '.lms103', lms_xformed)
        if os.path.exists(pathfolderfile_src[:-3] + 'npy'):
            shutil.copyfile(pathfolderfile_src[:-3] + 'npy', pathfolderfile_dst[:-3] + 'npy')
