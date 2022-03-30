from collections import OrderedDict

all_forward_dict = OrderedDict()
class mygraph(object):
    def __init__(self,):
        self.dict = OrderedDict()
        
graph = mygraph()
if __name__ == '__main__':
    g = mygraph()
    g.dict['0'] = 111
    print(g.dict)