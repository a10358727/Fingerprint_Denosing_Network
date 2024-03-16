
from trainer import trainer,tester
from options import options 


if __name__ == '__main__':

    print('start cvf')
    opt = options.Options().parse()     # Get the options
    if opt.train:                        # Train or test  
        trainer(opt)
    else:
        tester(opt)
