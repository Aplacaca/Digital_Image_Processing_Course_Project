from collections import OrderedDict


class mygraph(object):
    def __init__(self,):
        self.dict = OrderedDict()
    def flush(self,):
        del self.dict
        self.dict = OrderedDict()


graph = mygraph()
all_forward_dict = OrderedDict()

if __name__ == '__main__':
    g = mygraph()
    g.dict['0'] = 111
    print(g.dict)
