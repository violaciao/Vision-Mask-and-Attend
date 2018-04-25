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
        self.emb_dim = 512
        self.hid_dim = 512

        # training setting
        self.epochs = 50
        self.batch_size = 20
        self.lr = 1e-4
        self.grad_clip = 1
        self.dropout = 0.5
        self.gpu = True
        self.teaching_ratio = 0.5
        self.finetune = True

        # MISC
        self.seed = 1111
        self.resume = False
        self.load_best = False
        self.repeat = 1


    def prepare_dirs(self):
        import os
        dirs = [self.data_dir, self.ckpt_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.system('mkdir -p %s' % d)
