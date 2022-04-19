import numpy as np
import random
from dataset import Tiny_vid
class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A
            # check_version(A.__version__, '1.0.3')  # version requirement
 
            self.transform = A.Compose([
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)],
                bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
 
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
        class_list = labels[:,0].astypy(int).tolist()       
        box_list = [labels[i,1:] for i in range(labels.shape[0])]
        import pdb;pdb.set_trace()
        labels = list(map(lambda x,y:[x,y],class_list,box_list))
        return im, labels
    
if __name__=='__main__':
    myset = Tiny_vid()
    im = myset.images[:10]
    labels = myset.ground_truth[:10]
    # labels_array = []
    # for label in labels:
    #     tmp = []
    #     tmp = np.hstack([np.array([label[0]]),label[1]])
    #     labels_array.append(tmp)
    # labels_array = np.array(labels_array)
    myaug = Albumentations()
    outx,outy = myaug(im,labels)
    import pdb;pdb.set_trace()