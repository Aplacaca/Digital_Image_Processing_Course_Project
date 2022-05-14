# coding:utf-8
import torch
import traceback


def exception_handler(train_func):
    """处理训练时发生的异常并保存模型"""

    def wrapper(*args, **kwargs):
        try:
            return train_func(*args, **kwargs)
        except BaseException as e:
            if not isinstance(e, KeyboardInterrupt):
                traceback.print_exc()

            torch.cuda.empty_cache()

    return wrapper
