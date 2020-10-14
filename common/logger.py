# -*- coding: utf-8 -*-


class Logger(object):
    def __init__(self, args,mute=False):
        self.args = args
        self.mute=mute

    def print_args(self):
        print("weight: ", self.args.weight)
        print("lr: ", self.args.lr)
        print("n_epoch: ", self.args.n_epoch)
        print("batch_size: ", self.args.batch_size)
        print("n_gen: ", self.args.n_gen)
        print("dataset: ", self.args.dataset)
        print("outdir: ", self.args.outdir)
        print("print_interval: ", self.args.print_interval)
        print("model_name: ", self.args.model_name)
        # added
        if self.mute:
            print("mute_rate: ", self.args.mute_rate)



    def print_log(self, epoch, it, train_loss, val_loss):
        print("epoch: {}, iter: {}, train_loss: {}, val_loss: {}".format(
            epoch, it, train_loss, val_loss,
        ))
