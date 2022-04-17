import numpy as np
import pandas as pd
import jittor as jt
import os
from os.path import join, getsize

from jittor.dataset import Dataset
from PIL import Image
import re
import pdb
from copy import deepcopy


class Tiny_vid(Dataset):
    def __init__(self, data_dir="./tiny_vid/", train=True, transform=None):
        super().__init__()
        if train:
            self.set_attrs(total_len=750)
        else:
            self.set_attrs(total_len=150)
        self.transform=transform
        self.images = []
        self.ground_truth = []
        self.train = train
        self.class_dict = {'bird':0.,'car':1.,'dog':2.,'lizard':3.,'turtle':4.}
        self.debug=0
        # download if not exists
        if not os.path.exists(data_dir):
            os.system("wget http://xinggangw.info/data/tiny_vid.zip")
            os.system("unzip tiny_vid.zip")
        #
        
        vid_list = os.listdir("./tiny_vid/")
        label_dir_list = []
        gt_txt_list = []
        for ll in vid_list:
            if re.match(r"\.", ll) is None:
                label_dir_list.append(ll)
            if re.match(r"(.*)(_gt\.txt)", ll) is not None:
                gt_txt_list.append(ll)
        # print(gt_txt_list)        
        for root, dirs, files in os.walk(r'./tiny_vid/'):
            # get labels
            # pdb.set_trace()
            for name in files:
                if name in gt_txt_list:
                    # get ground truth
                    gt_name = str(re.match(r'(.*)(\.txt)',name).group(1))
                    exec(f"{gt_name}_fp = open(root+name)")
                    fd_lines = []
                    gt_lines = []
                    fd_lines = eval(f"{gt_name}_fp.readlines()")
                    exec(f"{gt_name}_fp.close()")
                    class_name = self.class_dict[str(re.match(r"(.*)(\_gt)",gt_name).group(1))]
                    for elem in fd_lines:
                        if len(gt_lines) == 180:
                            break
                        hh = elem.strip("\n")
                        hh = hh.split(" ")[1:]
                        hh = list(map(float, hh)) 
                        hh = list(map(lambda x: x/128, hh))
                        hh = [class_name, np.stack(hh)]
                        gt_lines.append(hh)
                        self.debug +=1
                    if self.train:
                        self.ground_truth.extend(gt_lines[0:150])
                        # print(len(gt_lines[0:150]))
                    else:
                        self.ground_truth.extend(gt_lines[150:180])
                        
                    exec(gt_name + "_lines" + "=" + "gt_lines[:180]")
                else:
                    # print(root)
                    # break
                    if name.endswith(".JPEG"): 
                        idx = int(str(re.match(r"(.*0*)(\.)", name).group(1)))
                        if idx > 180:
                            break
                        if self.train:
                            if idx > 150:
                                break
                        else:
                            if idx < 150:
                                continue

                        img = Image.open(join(root, name))
                        self.images.append(img)
        
                

        
        
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.images[index], self.ground_truth[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        
        return img, target
    
    
if __name__=='__main__':    
    dataset = Tiny_vid().set_attrs(batch_size=5, shuffle=True)
    # dataset = Tiny_vid()
    # print(len(dataset.ground_truth))
    for x, y in dataset:
        pdb.set_trace()
        print(x,y)