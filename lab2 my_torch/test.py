# from time import time


# def cal_time(func: callable):
#     def wrapper(*args, **kwargs):
#         start = time()
#         func(*args,**kwargs)
#         end = time()
#         print(f"consume {end - start}s")
#         return
#     return wrapper


# @cal_time
# def echo(str="HELLO WORLD"):
#     print(str)
#     return 8
    
# aa = echo()
# print(aa)

from visdom import Visdom

viz = Visdom(env="test")
assert viz.check_connection()

try:
    import matplotlib.pyplot as plt
    plt.plot([1, 23, 2, 4])
    plt.ylabel('some numbers')
    viz.matplot(plt)
except BaseException as err:
    print('Skipped matplotlib example')
    print('Error message: ', err)