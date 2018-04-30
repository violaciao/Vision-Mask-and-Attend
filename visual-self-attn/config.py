class Config(object):

    def __init__(self, 
            data_dir, 
            ckpt_dir):

        # data setting
        self.data_dir = data_dir
        self.ckpt_dir = ckpt_dir
        self.prepare_dirs()

        # model setting
        self.num_heads = 8
        self.num_layers = 6
        self.d_k = 64
        self.d_v = 64
        self.hid_dim = 128
        self.mode = 'mean'

        # training setting
        self.epochs = 50
        self.batch_size = 32
        self.lr = 1e-4
        self.grad_clip = 1
        self.dropout = 0.5
        self.gpu = True

        # MISC
        self.seed = 1111
        self.resume = False
        self.load_best = False


    def prepare_dirs(self):
        import os
        dirs = [self.data_dir, self.ckpt_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.system('mkdir -p %s' % d)
