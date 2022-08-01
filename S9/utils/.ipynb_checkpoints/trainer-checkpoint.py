import torch
import torch.nn as nn
# from model import get_model
from model.resnet import ResNet18
import torch.nn.functional as F
from tqdm import tqdm
import logging

class CfarTrainer:
    def __init__(self, **kwargs):
        super(CfarTrainer, self).__init__(**kwargs)
        self.logger = logging.getLogger(__name__)

    def build(self, config, output_dir, gpu):
        # initialize model
        self.output_dir = output_dir
        self.device = torch.device("cuda" if gpu else "cpu")
        print (self.device)
        # self.model = get_model(config['model']).to(self.device)
        self.model = ResNet18().to(self.device)

        target_layer = [self.model.layer4[-1]]
        # print("model ",self.model)
        self.logger.info(target_layer)

        # initialize loss function
        loss_config = config['loss']
        Loss = getattr(torch.nn, loss_config.pop('name'))
        self.loss_func = Loss(**loss_config)

        # initialize optimizer
        optimizer_config = config['optimizer']
        Optim = getattr(torch.optim, optimizer_config.pop('name'))
        self.optimizer = Optim(self.model.parameters(), **optimizer_config)

        # initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20,40], gamma=0.2)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        # initialize logger
        self.logger.info(self.model)
        self.logger.info('Number of parameters: %i',
                             sum(p.numel() for p in self.model.parameters()))
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        summary = dict()
        correct=0
        processed=0
        train_l=0
        train_a=0
        self.model.train()

        correct=0
        processed=0
        pbar = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            # loss = F.nll_loss(output, target)
            loss = self.loss_func(output, target)
            train_l = loss
            # train_loss.append(loss)
            loss.backward()
            self.optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx} Accuracy={100*correct/processed}')
            train_a = (100*correct/processed)

            # Return training loss and accuracy
        self.scheduler.step()
        return (train_l, train_a)

    @torch.no_grad()
    def evaluate(self, test_loader, use_gradcam):
        # global prev_val_acc
        """"Evaluate the model"""

        self.model.eval()
        print("\n use gradcam or not ",use_gradcam)

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                print(data.shape)
                print(target)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_func(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        # valid_loss = sum_loss / (i + 1)
        self.logger.debug('Processed %i samples ', len(test_loader.sampler))
        self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        val_acc = correct / len(test_loader.dataset)
        if (val_acc > self.prev_val_acc):
            prev_val_acc = val_acc
            torch.save(self.model, self.output_dir+'/{}'.format('model.pth'))

        # Return test loss and accuracy
        return (test_loss, val_acc)

    out_train_acc = {}
    out_train_loss = {}
    out_val_acc = {}
    out_val_loss = {}
    total_epochs = {}

    def train(self, train_data, valid_data, n_epochs, use_gradcam):
        for epoch in range(1, n_epochs):
            self.prev_val_acc = 0
            self.total_epochs[epoch] = []
            self.out_train_acc[epoch] = []
            self.out_train_loss[epoch] = []
            self.out_val_acc[epoch] = []
            self.out_val_loss[epoch] = []
            self.logger.info('Epoch %i', epoch)
            summary = dict(epoch=epoch)
            train_loss, train_acc = self.train_epoch(train_data)
            valid_loss, val_acc = self.evaluate(valid_data, use_gradcam)
            self.out_train_acc[epoch].append(train_acc)
            self.out_train_loss[epoch].append(train_loss)
            self.out_val_acc[epoch].append(val_acc)
            self.out_val_loss[epoch].append(valid_loss)
            summary.update(train_loss=train_loss, train_acc=train_acc,
                            valid_loss=valid_loss, valid_acc=val_acc)
            self.logger.info('Epoch %i summary: %s', epoch, summary)
            self.logger.info('\n')
            print('Epoch: %i, Train Loss: %.3f, Valid Loss: %.3f' % (epoch, train_loss, valid_loss))
            self.logger.info('Epoch: %i, Train Loss: %.3f, Valid Loss: %.3f', epoch, train_loss, valid_loss)
            


def get_trainer(**kwargs):
    return CfarTrainer(**kwargs)
