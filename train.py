from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import torch
import torch.utils.data
import torch.utils.data
from torch.nn import functional as F
from torch.autograd import Variable
import wandb
from torch import nn
from torch import optim
from pytorchtools import EarlyStopping
from tqdm import tqdm


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


class Train():
    def __init__(self, num_tasks, batch_size=64, alpha=0.5, n_epochs=100, nb_class=10, patience=10, lr=1e-3,
                 weight_decay=1e-5):
        self.max_epoch = n_epochs
        self.nb_class = nb_class
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.d = Data(self.batch_size)
        self.alpha = alpha
        self.alpha_list = [self.alpha / 2, self.alpha]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def _mask(self, curr_tensor, prev_tensor, p):
        binomial = torch.distributions.binomial.Binomial(probs=p)
        mask = binomial.sample(curr_tensor.size())
        mask = mask.to(self.device)
        curr=nn.Parameter(curr_tensor * mask)
        prev=nn.Parameter(prev_tensor * (1-mask))
        return curr+prev
    
    def _copyLayer(self, past_model, curr_model, p):
        '''
        Drop transfer
        After training for (t-1)th task, the weights and bias are transfered
        to the new model for (t)th task with probability p
        '''
        past_params = dict()
        pp = past_model.state_dict()
        curr_params = dict()
        cp = past_model.state_dict()

        trans_model = Net()
        d = trans_model.state_dict()
        for n, p in pp.items():
            past_params[n] = p
            curr_params[n] = cp[n]
            d[n] = self._mask(curr_params[n], past_params[n], 0.5)
        trans_model.load_state_dict(d)
        return trans_model
    
    def train(self, model, task_idx=0, past_model=None, fisher_matrix_sample_size=640, wandb_log=False,
              consolidate=False):
        # to track the training loss as the model trains
        train_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # initialize the early_stopping object
        if task_idx == 0:
            print("task_idx ",task_idx)
            self.train_loader = self.d.train_loader
            self.valid_loader = self.d.valid_loader
            self.early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        elif task_idx == 1:
            print("task_idx ",task_idx)
            self.train_loader = self.d.trainA_loader
            self.valid_loader = self.d.validA_loader
            self.early_stopping = EarlyStopping(patience=self.max_epoch, verbose=True)
        elif task_idx == 2:
            print("task_idx ",task_idx)
            self.train_loader = self.d.trainB_loader
            self.valid_loader = self.d.validB_loader
            self.early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        
        self.test_loader = self.d.test_loader
        model.to(self.device)
        if past_model != None:
            past_model.to(self.device)
        
        for epoch in range(1, self.max_epoch + 1):

            ###################
            # train the model #
            ###################
            running_loss = 0.0
            total = 0
            correct = 0
            for batch, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                ce_loss, e_loss = criterion(output, target), 0
                if consolidate:
                    e_loss = model.ewc_loss()
                loss = ce_loss + e_loss
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                train_loss = np.average(train_losses)
                avg_train_losses.append(train_loss)


                # Predictions
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()


            # Calculate global accuracy
            try:
                accuracy = 100 * correct / total
                if wandb_log:
                    wandb.log({"accuracy": accuracy})

            except:
                pass

            if consolidate:
                self.fisher_estimation_sample_size = fisher_matrix_sample_size
                model.consolidate(model.estimate_fisher(
                    self.train_loader, self.fisher_estimation_sample_size, self.batch_size
                ))

            
            # clear lists to track next epoch
            train_losses = []
            es, avg_valid_losses = self.evaluate(model, epoch, wandb_log)
            if es.early_stop:
                print("Early stopping")
                break
        model.load_state_dict(torch.load('checkpoint.pt'))

        return model, avg_train_losses, avg_valid_losses

    def evaluate(self, model, epoch, wandb_log=False):
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []
        model.eval()  # prep model for evaluation
        correct = 0
        total = 0
        confusion_matrix = torch.zeros(self.nb_class, self.nb_class)
        for data, target in self.valid_loader:
            data, target = data.to(self.device), target.to(self.device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())
            # Predicted results
            _, preds = torch.max(output, 1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            for t, p in zip(target, preds):
                confusion_matrix[t, p] += 1

        # Model Performance Statistics
        accuracy = 100 * correct / total

        TP, TN, FP, FN, precision, recall, f1 = [np.zeros(self.nb_class, dtype='float64')] * 7

        # Normalize confusion matrix
        for i in range(self.nb_class):
            TP[i] = confusion_matrix[i][i]
            TN[i] = confusion_matrix[-i][-i]
            FP[i] = confusion_matrix[i][-i]
            FN[i] = confusion_matrix[-i][i]
            precision[i] = TP[i] / (TP[i] + FP[i])
            recall[i] = TP[i] / (TP[i] + FN[i])
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum()

        # print training/validation statistics
        # calculate average loss over an epoch

        print("f1 vector : ", f1)
        macro_f1 = reduce(lambda a, b: a + b, f1) / len(f1) * 100
        precision = reduce(lambda a, b: a + b, precision) / len(precision) * 100
        recall = reduce(lambda a, b: a + b, recall) / len(recall) * 100

        print("Macro f1 score is {:.3f}%".format(macro_f1))
        print('Accuracy of the model {:.3f}%'.format(accuracy))

        valid_loss = np.average(valid_losses)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(self.max_epoch))

        print(f'[{epoch:>{epoch_len}}/{self.max_epoch:>{epoch_len}}] ' +
                     f'valid_loss: {valid_loss:.5f}')

        if wandb_log:
            wandb.log({"Precision(valid)": precision, "Recall(valid)": recall, "F1 Score(valid)": macro_f1,
                       "Accuracy(valid)": accuracy})
        self.early_stopping(valid_loss, model)
        
        return self.early_stopping, avg_valid_losses

    def train_drop_transfer(self, consolidate, P=5, M=0.6, E="1:1",wandb_log = True):
        for t in range(1,self.num_tasks+1):
            net = Net()
            net.to(self.device)

            config = dict(
                learning_rate=0.001,
                weight_decay=0.00001,
                epoch=self.max_epoch,
                momentum=0.9,
                architecture="CNN",
                dataset_id="cifar-10",
                batch_size=self.batch_size,
                early_stopping=self.patience
            )

            if t == 1:
                print("🥰 Start Training First Task")
                new_net, avg_train_losses, avg_valid_losses = self.train(net, task_idx=t)
            

            # Drop Transfer
            if t>1:
                wandb.init(
                  project='Seq Boost',
                  config=config,
                  name='SeqBoost task {} + drop transfer p={} mu={} eta={}'.format(t, P, M, E))

                print("🥰 Start Training Second Task")
                past_net = Net()
                past_net.eval()
                new = new_net.state_dict()
                past_net.load_state_dict(new)
                past_net.to(self.device)
                
                net = self._copyLayer(past_net, net, 0.5)
                net.to(self.device)
                new_net, avg_train_losses, avg_valid_losses = self.train(net, task_idx=t, past_model = past_net, consolidate=consolidate, wandb_log = wandb_log)
                

    def test(self, model, wandb_log=True, class_names=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                                                'ship', 'truck')):
        model.eval()
        correct = 0
        total = 0
        confusion_matrix = torch.zeros(self.nb_class, self.nb_class)

        with torch.no_grad():
            for data, target in self.test_loader:
                if len(target.data) != self.batch_size:
                    break
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += self.batch_size
                correct += (predicted == target).sum().item()
                for i in range(self.batch_size):
                    label = target.data[i]
                for t, p in zip(target, predicted):
                    confusion_matrix[t, p] += 1

        # Calculate global accuracy
        if wandb_log:
            try:
                accuracy = 100 * correct / total
                wandb.log({"accuracy": accuracy})
            except:
                pass

        # F1 score 직접 계산하기
        print(confusion_matrix)
        class_correct = confusion_matrix.diag()
        class_total = confusion_matrix.sum(1)
        class_accuracies = class_correct / class_total

        TP, TN, FP, FN, precision, recall, f1 = [np.zeros(self.nb_class, dtype='float64')] * 7

        # Normalize confusion matrix
        for i in range(self.nb_class):
            TP[i] = confusion_matrix[i][i]
            TN[i] = confusion_matrix[-i][-i]
            FP[i] = confusion_matrix[i][-i]
            FN[i] = confusion_matrix[-i][i]
            precision[i] = TP[i] / (TP[i] + FP[i])
            recall[i] = TP[i] / (TP[i] + FN[i])
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum()

        # Print statistics
        macro_f1 = reduce(lambda a, b: a + b, f1) / len(f1) * 100
        precision = reduce(lambda a, b: a + b, precision) / len(precision) * 100
        recall = reduce(lambda a, b: a + b, recall) / len(recall) * 100
        print("Macro f1 score is {:.3f}%".format(macro_f1))
        try:
            print('Accuracy of the model {:.3f}%'.format(accuracy))
        except:
            pass
        for i in range(self.nb_class):
            print('Accuracy for {}: {:.3f}%'.format(
                class_names[i], 100 * class_accuracies[i]))
            if wandb_log:
                wandb.log({f"Accuracy of class {class_names[i]}": class_accuracies[i] * 100})
        if wandb_log:
            try:
                wandb.log(
                    {"Precision(test)": precision, "Recall(test)": recall, "F1 Score(test)": macro_f1,
                     "Accuracy(test)": accuracy})
            except:
                wandb.log(
                    {"Precision(test)": precision, "Recall(test)": recall, "F1 Score(test)": macro_f1})