from inspect import getsource


def help(opt):
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python {0} <function> [--args=value,]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    source = (getsource(opt.__class__))
    print(source)
