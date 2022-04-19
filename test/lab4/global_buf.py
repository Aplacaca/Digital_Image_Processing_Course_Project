import numpy as np
# import pdb

global_list_img = np.load("./transformed_img.npy")
global_list_label = np.load("./transformed_label.npy")

class_list = global_list_label[:,0].astype(int).tolist()       
box_list = [global_list_label[i,1:] for i in range(global_list_label.shape[0])]

global_labels = list(map(lambda x,y:[x,y],class_list,box_list))
global_imgs =  [x for x in global_list_img]
# pdb.set_trace()


# import pdb
# myset = BUF()
# my_aug = MyAlbumentations()
# img = deepcopy(myset.images) 
# label = deepcopy(myset.ground_truth)
# global_list_img, global_list_label = my_aug(img,label)
# # pdb.set_trace()
# a = np.array(global_list_img)
# al = np.array(global_list_label)
# np.save("./transformed_img.npy",a)
# np.save("./transformed_label.npy",al)