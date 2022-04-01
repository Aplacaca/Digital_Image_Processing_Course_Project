from time import time


def cal_time(func: callable):
    def wrapper(*args, **kwargs):
        start = time()
        func(*args,**kwargs)
        end = time()
        print(f"consume {end - start}s")
        return
    return wrapper


@cal_time
def echo(str="HELLO WORLD"):
    print(str)
    return 8
    
aa = echo()
print(aa)