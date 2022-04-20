import numpy as np
# import pdb

global_list_img = np.load("./transformed_img1.npy")
global_list_label = np.load("./transformed_label1.npy")

class_list = global_list_label[:,0].astype(int).tolist()       
box_list = [global_list_label[i,1:] for i in range(global_list_label.shape[0])]

global_labels = list(map(lambda x,y:[x,y],class_list,box_list))
global_imgs =  [x for x in global_list_img]
# pdb.set_trace()


