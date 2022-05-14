import numpy as np
import random
import albumentations as Alb

class MyAlbumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            self.transform = Alb.Compose([
                Alb.OneOf([
                Alb.Blur(p=0.1),
                Alb.MedianBlur(p=0.1),
                Alb.ToGray(p=0.1),
                Alb.ImageCompression(quality_lower=75, p=0.1)
                ],p = 0.2),
                Alb.CLAHE(p=0.1),
                Alb.RandomBrightnessContrast(p=0.1),
                Alb.RandomGamma(p=0.1),
                Alb.RandomCrop(height=128,width=128,p=0.7),
                Alb.HorizontalFlip(p=0.3),
                Alb.VerticalFlip(p=0.3),
                Alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3)],
                # 随机应用仿射变换：平移，缩放和旋转输入
                bbox_params=Alb.BboxParams(format='albumentations', label_fields=['class_labels']))
 
            # logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            pass
            # logging.info(colorstr('albumentations: ') + f'{e}')
 
    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=[labels[1].tolist()], class_labels=[labels[0]])  # transformed
            # im, labels = new['image'], np.array([new['class_labels'], new['bboxes']])
            im, label_s = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
            # pdb.set_trace()
            if label_s.shape[0] != 0:
                labels = label_s.squeeze(0)
            # else:
            #     labels = 
        return im, labels
    
if __name__=='__main__':
    import pdb
    from dataset import Tiny_vid
    from copy import deepcopy
    myaug = MyAlbumentations()
    
    myset = Tiny_vid(aug=False)
    my_aug = MyAlbumentations()
    # pdb.set_trace()
    imgs = deepcopy(myset.images) 
    labels = deepcopy(myset.ground_truth)
    # import cv2
    # im = cv2.imread("./tiny_vid/bird/000001.JPEG")
    global_list_img = []
    global_list_label = []
    
    for img, label in zip(imgs,labels):
        out_img, out_label_list = my_aug(im=img.transpose(1,2,0),labels=label)
        global_list_img.append(out_img.transpose(2,0,1))
        global_list_label.append(out_label_list)
        if len(global_list_img)%50 == 0:
            print(len(global_list_img))
    
    
    a = np.array(global_list_img)
    al = np.array(global_list_label)
    
    np.save("./transformed_img1.npy",a)
    np.save("./transformed_label1.npy",al)
    pdb.set_trace()
    # print(a.shape)
    # print(al.shape)
    
    # pdb.set_trace()
    
    # img2 = deepcopy(myset.images) 
    # label2 = deepcopy(myset.ground_truth)
    # global_list_img2, global_list_label2 = my_aug(img2,label2)
    
    pdb.set_trace()