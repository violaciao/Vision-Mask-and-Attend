import sys

from config import Config
from trainer import Trainer


def main():
    data_dir = sys.argv[1]
    ckpt_dir = sys.argv[2]
    mode = sys.argv[3]

    config = Config(data_dir, ckpt_dir)
    trainer = Trainer(config)
    if mode == 'train':
        trainer.train()
    elif mode == 'eval':
        e, best_acc = trainer.load_ckpt(True)
        print('best acc = {:2.2f} @epoch {}'.format(best_acc, e))
        loss, cmat, auc  = trainer.eval(trainer.valid_dataloader)
        print('loss:\t{:2.4f}\nacc:\t{:2.2f}'.format(loss, 100 * cmat.acc))
        print(cmat)
        print(auc)


if __name__ == '__main__':
    main()
