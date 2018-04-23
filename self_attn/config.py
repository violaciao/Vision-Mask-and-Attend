class Config(object):

    def __init__(self, 
            data_dir, 
            ckpt_dir):

        # data setting
        self.num_classes = 5
        self.data_dir = data_dir
        self.ckpt_dir = ckpt_dir
        self.prepare_dirs()

        # model setting
        self.num_heads = 8
        self.num_layers = 6
        self.d_k = 75
        self.d_v = 75
        self.hid_dim = 128
        self.feature_maps = (12, 24) # 25k == num_heads x d_k
        self.fc_hids = [1024, 256]

        # training setting
        self.epochs = 20
        self.batch_size = 10
        self.lr = 1e-5
        self.dropout = 0.3
        self.gpu = False

        # MISC
        self.seed = 1111
        self.resume = True


    def prepare_dirs(self):
        import os
        dirs = [self.data_dir, self.ckpt_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.system('mkdir -p %s' % d)
