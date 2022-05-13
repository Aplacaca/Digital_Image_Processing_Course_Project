import os
import torch
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt


def bbox_to_rect(bbox, color):
    """Defined in :numref:`sec_bbox`"""
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """
    显示所有边界框

    Params:
    ------------
    bboxes: list[tensor]
        边界框 `[N*(x1, y1, x2, y2)]`
    labels: list[str]
        标签字符 `[str, str, ……]`
    """

    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])

    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().cpu().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center',
                      ha='center', fontsize=9, color=text_color, bbox=dict(facecolor=color, lw=0))


def select_pred(img, pred, target, image_size=300, num_classes=5, top_m=3, device='cuda:0'):
    """选择并打印含top_m个检测框的图片

    Parameters:
    ------- 
        img: tensor
            图片 `[C, H, W]`
        pred: tensor
            检测框 `[num_classes, top_k, 5]`，坐标(x1, y1, x2, y2) + 置信度
        target: tensor
            目标框 `[1, 5]`，坐标(x1, y1, x2, y2) + 置信度
        image_size: int
            图片大小
        num_classes: int
            类别数量
        top_m: int
            打印检测框的数量（不超过num_classes）

    Returns:
    --------
        bbox_best: tensor
            最佳的检测框 `[top_m, 4]`
        conf_best:
            最佳的类别置信度 `[top_m]`
        best_k:
            最佳的类别索引 `[top_m]`

    """

    # 取预测概率最高的 top_m个不同类别的预测框信息
    top_m = min(top_m, num_classes)  # 打印检测框最多有num_classes个
    # 各类别按概率从大到小排top_k个，虽然每类别最多有top_k个预测框，但只取最大概率的来比较
    best_k = torch.sort(pred[:, 0, -1], dim=0, descending=True)[1][:top_m]
    bbox_best = pred[:, 0, :-1][best_k]
    conf_best = pred[:, 0, -1][best_k]

    # 打印最终图片
    img = transforms.ToPILImage()(img.clone().squeeze(0))
    fig = plt.imshow(img)
    title = ('bird', 'car', 'dog', 'lizard', 'turtle')[int(target[0][-1])-1]
    plt.title(title)
    bbox_scale = torch.tensor([image_size]*4, device=device)
    show_bboxes(fig.axes, [target[0][:-1].to(device) *
                bbox_scale], labels=title, colors='black')

    bboxes = []
    labels = []
    for i, class_id in enumerate(best_k):
        labels.append(('bird=', 'car=', 'dog=', 'lizard=', 'turtle=')[
                      class_id-1] + f'{conf_best[i]*100:.2f}%')
        bboxes.append(bbox_best[i] * bbox_scale)
    show_bboxes(fig.axes, bboxes, labels)
    plt.show()

    return bbox_best, conf_best, best_k


def plot_PR(num_classes, metric, overlap_thresh):
    """绘制所有类别的PR曲线

    Parameters:
    ------- 
        num_classes: int
            类别数
        metric: list[tuple]
            每个类别的测量记录，[num_classes, (P, R, AP)]

    """

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['figure.figsize'] = (15.0, 10.0)  # 设置figure_size尺寸

    class_names = ('bird', 'car', 'dog', 'lizard', 'turtle')
    colors = ['r', 'b', 'g', 'm', 'y']

    for class_id in range(num_classes):
        P = metric[class_id][0]
        R = metric[class_id][1]
        name = class_names[class_id]

        plt.subplot(2, 3, class_id+1)
        plt.title(
            f'PR Curve | AP@{int(100*overlap_thresh):d}: {100*metric[class_id][2]:.2f}%')
        plt.plot(R, P, color=colors[class_id], ms=10,
                 label=name)  # "r" 表示红色，ms用来设置*的大小

        # 将图例显示到左上角
        plt.legend(loc="upper right")
        # 设置坐标轴
        plt.xlabel("Recall")
        plt.ylabel("Precise")

    if not os.path.exists("./eval/"):
        os.mkdir("./eval/")
    plt.savefig('./eval/PR_' + str(int(100*overlap_thresh)) + '.jpg')
    # plt.show()
