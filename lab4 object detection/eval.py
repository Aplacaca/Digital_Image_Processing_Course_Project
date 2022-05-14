# coding:utf-8
import ipdb
import torch

from ssd import SSD
from utils.visdom import Visualizer
from utils.visualize import plot_PR
from utils.visualize import select_pred
from utils.random_seed import setup_seed
from utils.box_utils import jaccard_overlap

setup_seed(729)
DISPLAY = False  # 逐张显示预测结果


class EvalPipeline:
    """ 测试模型流水线 """

    def __init__(self, model_path: str, num_classes=5, dataloader=None, image_size=300, top_k=10,
                 conf_thresh=0.05, overlap_thresh=0.5, use_gpu=True):
        """
        Parameters
        ----------
        model_path: str
            模型文件路径

        dataloader:
            测试集

        image_size: int
            图像尺寸

        top_k: int
            一张图片中每一个类别最多保留的预测框数量

        conf_thresh: float
            置信度阈值

        overlap_thresh: float
            IOU 阈值

        use_gpu: bool
            是否使用 GPU
        """
        self.top_k = top_k
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.dataloader = dataloader
        self.image_size = image_size
        self.conf_thresh = conf_thresh
        self.overlap_thresh = overlap_thresh
        self.device = 'cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu'

        # 保存每个类别的预测结果，[num_classes * (图片数, (conf, tp, fp))]
        self.record = [[] for _ in range(self.num_classes)]

        self.model_path = model_path
        self.model = SSD(
            self.num_classes+1,
            top_k=top_k,
            image_size=image_size
        )
        self.model = self.model.to(self.device)
        self.model.load(model_path, device=self.device)
        self.model.eval()

        self.vis = Visualizer('Eval')

    @torch.no_grad()
    def eval(self) -> float:
        """ 测试模型，获取 mAP """
        self._predict()
        mAP = self.metric()
        torch.cuda.empty_cache()

        return mAP

    def _predict(self):
        """ 预测每一种类存在于哪些图片中 """

        print('🛸 正在预测中...')
        for images, targets in self.dataloader:

            # 预测
            images = images.to(self.device)
            out = self.model.predict(images)

            # 展示每张图片的top_m最佳预测框
            if DISPLAY:
                for batch_idx in range(len(images)):
                    # 选择分类概率最大的top_m=1个预测结果
                    bbox_best, conf_best, class_best = select_pred(self.vis, images[batch_idx], torch.tensor(
                        out[batch_idx]).to(self.device), targets[batch_idx], top_m=1)

            # 记录每个类别的预测结果
            self._record(out, targets)

    def metric(self):
        metric = [self._get_AP(i) for i in range(self.num_classes)]

        # 绘制PR曲线图
        plot_PR(self.num_classes, metric, self.overlap_thresh)

        # 计算mAP
        mAP = 0
        for item in metric:
            mAP += item[-1]
        mAP /= len(metric)

        return mAP

    def _get_AP(self, class_id):
        """ 计算一个类的预测效果

        Parameters
        ----------
        class_id: str
            类别号，(0 ~ num_classes-1)

        Returns
        -------
        ap: float
            AP，没有预测出这个类就返回 0

        iou: float
            平均交并比

        precision: list
            查准率

        recall: list
            查全率
        """

        # 读取
        record = torch.tensor(self.record[class_id], device=self.device)
        index = torch.sort(record[:, 0], descending=True)[1]
        record = record[index]

        # 计算 TP、FP、Precise、Recall
        tp = record[:, 1].cpu().numpy().cumsum()
        fp = record[:, 2].cpu().numpy().cumsum()
        P = tp / (tp+fp)
        R = tp / len(record)

        # 计算 AP
        AP = 0
        index = [0]
        for i in range(len(P)-1):
            if P[i] > P[i+1]:
                index.append(i)
        for i in range(1, len(index)):
            if i == 0:
                AP += P[index[i]] * R[index[i]]
            else:
                AP += P[index[i]] * (R[index[i]] - R[index[i-1]])

        return P, R, AP

    def _record(self, preds, targets, ):
        """记录每个图片的预测conf、tp、fp

        Parameters:
        ------- 
            preds:
                每个batch的预测结果 `[batch_size, 6, top_k, 5]`

            targets:
                每个batch的实际标签 `[batch_size, 1, 5]`，这里只适用于图片含1个目标框的情况

        """

        for i in range(preds.shape[0]):
            pred = preds[i]
            true_class = int(targets[i, 0, -1].item())  # 图片真实类别

            # 计算每个预测框与目标框的IOU
            bbox = pred[:, :, :-1].reshape(-1, 4)
            iou = jaccard_overlap(bbox, targets[i, :, :-1])
            iou = iou.reshape(preds.shape[1], preds.shape[2], 1)

            # 选出IOU最大的预测框
            max_i = int(iou.argmax() / preds.shape[2])  # 对应行数，也是class_id
            max_j = iou.argmax() % preds.shape[2]

            # 记录此图片的conf、tp、fp
            record = []
            record.append(pred[max_i, max_j, -1].item())
            if max_i != true_class:
                record.append(0)
                record.append(1)
            else:
                record.append(1 if iou[max_i, max_j] >=
                              self.overlap_thresh else 0)
                record.append(1 if iou[max_i, max_j] <
                              self.overlap_thresh else 0)

            # 保存
            self.record[true_class-1].append(record)


if __name__ == "__main__":
    # load dataset
    from dataset.data import load_data
    num_classes = 5
    batch_size = 16
    _, test_loader = load_data(batch_size)

    # eval
    model_path = 'best.pth'
    eval_pipeline = EvalPipeline(
        model_path, num_classes, test_loader, conf_thresh=0.001, overlap_thresh=0.5)
    mAP = eval_pipeline.eval()
