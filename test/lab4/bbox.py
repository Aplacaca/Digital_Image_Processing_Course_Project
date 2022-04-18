import numpy as np
import matplotlib.pyplot as plt
import jittor as jt
import pdb
from dataset import Tiny_vid

def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    inter_upperlefts = jt.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = jt.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts)
    inters = jt.clamp(inters, min_v=0.0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def box_iou_batch(boxes1, boxes2):
    """计算两个锚框batch的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    inter_upperlefts = jt.maximum(boxes1[:, :2], boxes2[:, :2])
    inter_lowerrights = jt.minimum(boxes1[:, 2:], boxes2[:, 2:])
    
    inters = (inter_lowerrights - inter_upperlefts)
    inters = jt.clamp(inters, min_v=0.0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, 0] * inters[:, 1]
    union_areas = areas1 + areas2 - inter_areas
    return inter_areas / union_areas


if __name__ == '__main__':
    te = Tiny_vid().set_attrs(batch_size=5, shuffle=True)
    i=0
    bb=[]
    for x, y in te:
        i+=1
        if i > 2:
            break
        bb.append(y)

    res=box_iou(bb[0][1],bb[1][1])
    pdb.set_trace() 
    