import sys

from config import Config
from trainer import Trainer


def main():
    data_dir = sys.argv[1]
    ckpt_dir = sys.argv[2]

    config = Config(data_dir, ckpt_dir)
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
