import torch
from torchsummary import summary
import yaml
import logging
import os

from dataset.dataset import get_data_loaders
from utils.trainer import CfarTrainer
from utils.logging import get_logger
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', nargs='?', default='configs/cfg.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, default='output', help='Path to output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--resume', action='store_true',help='Resume training from last checkpoint')
    return parser.parse_args()

# load config function
def load_config(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():

    args = parse_args()
    config = load_config(args)
    output_dir = config.get('output_dir', None)
    if output_dir is not None:
        output_dir = os.path.expandvars(output_dir)
        os.makedirs(output_dir, exist_ok=True)


    # Set up logging
    log_file = (os.path.join(output_dir, 'out.log')
                if output_dir is not None else None)
    print("verbose ", args.verbose)
    get_logger(verbose=args.verbose, log_file=log_file, append=args.resume)
    logging.info('Initialized logging')

    #get the GPU
    gpu = args.gpu
    batch_size = config['data']
    print(batch_size['batch_size'])
    train_data_loader, valid_data_loader, classes = get_data_loaders(batch_size['batch_size'], gpu)

    # create trainer class object
    trainer = CfarTrainer()

    #get epochs from config
    epochs = config.get('train', 10)
    print(epochs['n_epochs'])

    # build model
    trainer.build(config, output_dir, gpu=gpu)

    # traninig loop and save model
    trainer.train(train_data_loader, valid_data_loader, epochs['n_epochs'])


    print("Finished Training")

if __name__ == '__main__':
    main()

