import numpy as np
import random
import albumentations as Alb

class MyAlbumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            self.transform = Alb.Compose([
                Alb.Blur(p=0.01),
                Alb.MedianBlur(p=0.01),
                Alb.ToGray(p=0.02),
                Alb.CLAHE(p=0.02),
                Alb.RandomBrightnessContrast(p=0.0),
                Alb.RandomGamma(p=0.0),
                Alb.ImageCompression(quality_lower=75, p=0.0),
                Alb.HorizontalFlip(p=0.5),
                Alb.VerticalFlip(p=0.5),
                Alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3)],
                # 随机应用仿射变换：平移，缩放和旋转输入
                bbox_params=Alb.BboxParams(format='coco', label_fields=['class_labels']))
 
            # logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            pass
            # logging.info(colorstr('albumentations: ') + f'{e}')
 
    def __call__(self, im, labels, p=1.0):
        labels_array = []
        im = np.array(im)
        for label in labels:
            tmp = []
            tmp = np.hstack([np.array([label[0]]),label[1]])
            labels_array.append(tmp)
        labels_array = np.array(labels_array)
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels_array[:,1:], class_labels=labels_array[:,0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        
        # class_list = labels[:,0].astype(int).tolist()       
        # box_list = [labels[i,1:] for i in range(labels.shape[0])]
        # labels = list(map(lambda x,y:[x,y],class_list,box_list))
        # import pdb;pdb.set_trace()
        return im, labels
    
if __name__=='__main__':
    # myset = Tiny_vid()
    # im = myset.images[:10]
    # labels = myset.ground_truth[:10]
    # # labels_array = []
    # # for label in labels:
    # #     tmp = []
    # #     tmp = np.hstack([np.array([label[0]]),label[1]])
    # #     labels_array.append(tmp)
    # # labels_array = np.array(labels_array)
    myaug = MyAlbumentations()
    # outx,outy = myaug(im,labels)
    import pdb;pdb.set_trace()