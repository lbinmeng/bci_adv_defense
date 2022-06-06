import os
import logging
import torch
import argparse
import attack_lib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from model import EEGNet, DeepConvNet, ShallowConvNet
from utils.data_loader import ERNLoad, MI4CLoad, EPFLLoad, split
from utils.pytorch_utils import bca_score, init_weights, print_args, seed, adjust_learning_rate, weight_for_balanced_classes


def train(x: torch.Tensor, y: torch.Tensor, x_test: torch.Tensor,
          y_test: torch.Tensor, model_save_path: str, args):
    # initialize the model
    if args.model == 'EEGNet':
        model = EEGNet(n_classes=len(np.unique(y.numpy())),
                       Chans=x.shape[2],
                       Samples=x.shape[3],
                       kernLenght=64,
                       F1=4,
                       D=2,
                       F2=8,
                       dropoutRate=0.25,
                       norm_rate=0.25).to(args.device)
    elif args.model == 'DeepCNN':
        model = DeepConvNet(n_classes=len(np.unique(y.numpy())),
                            Chans=x.shape[2],
                            Samples=x.shape[3],
                            dropoutRate=0.5).to(args.device)
    elif args.model == 'ShadowCNN':
        model = ShallowConvNet(n_classes=len(np.unique(y.numpy())),
                               Chans=x.shape[2],
                               Samples=x.shape[3],
                               dropoutRate=0.5).to(args.device)
    else:
        raise 'No such model!'
    model.apply(init_weights)

    # trainable parameters
    params = []
    for _, v in model.named_parameters():
        params += [{'params': v, 'lr': args.lr}]

    optimizer = optim.Adam(params)
    # normal train loss
    criterion_cal = nn.CrossEntropyLoss().to(args.device)
    criterion_kl = nn.KLDivLoss(reduction='batchmean').to(args.device)

    # data loader
    # for unbalanced dataset, create a weighted sampler
    sample_weights = weight_for_balanced_classes(y)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))
    train_loader = DataLoader(dataset=TensorDataset(x, y),
                              batch_size=args.batch_size,
                              sampler=sampler,
                              num_workers=1,
                              drop_last=False)
    test_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=1,
                             drop_last=False)

    for epoch in range(args.epochs):
        # model train
        adjust_learning_rate(optimizer=optimizer,
                             epoch=epoch + 1,
                             learning_rate=args.lr)
        model.train()
        l = 0.
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            optimizer.zero_grad()
            # optimize on robust loss
            if epoch <= 20:
                batch_adv_x = batch_x
            else:
                batch_adv_x = attack_lib.PGD_batch(model,
                                                   batch_x,
                                                   batch_y,
                                                   eps=0.05,
                                                   alpha=0.005,
                                                   steps=10)

            model.train()
            optimizer.zero_grad()

            adv_logits = model(batch_adv_x)
            logits = model(batch_x)

            loss_cal = criterion_cal(logits, batch_y)
            loss_rob = criterion_kl(F.log_softmax(adv_logits, dim=1),
                                    F.softmax(logits, dim=1))
            loss = loss_cal + args.beta * loss_rob
            loss.backward()
            optimizer.step()

            model.MaxNormConstraint()
            l += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            train_loss, train_acc, _ = eval(model, criterion_cal, train_loader)
            test_loss, test_acc, _ = eval(model, criterion_cal, test_loader)

            logging.info(
                'Epoch {}/{}: train loss: {:.4f} train acc: {:.2f} | test loss: {:.4f} test acc: {:.2f}'
                .format(epoch + 1, args.epochs, train_loss, train_acc,
                        test_loss, test_acc))
    torch.save(model.state_dict(), model_save_path + '/model.pt')

    model.eval()
    _, test_acc, test_bca = eval(model, criterion_cal, test_loader)
    logging.info(f'test bca: {test_bca}')

    # attack using PGD and FGSM
    adv_accs = []
    for eps in [0.01, 0.05, 0.1]:
        adv_x = attack_lib.PGD(model,
                               x_test,
                               y_test,
                               eps=eps,
                               alpha=eps / 10,
                               steps=10)
        adv_loader = DataLoader(dataset=TensorDataset(adv_x.cpu(), y_test),
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=1,
                                drop_last=False)
        _, adv_acc, _ = eval(model, criterion_cal, adv_loader)
        adv_accs.append(adv_acc)
        logging.info(f'test acc: {test_acc}, PGD {eps} adv acc: {adv_acc}')
    for eps in [0.01, 0.05, 0.1]:
        adv_x = attack_lib.FGSM(model, x_test, y_test, eps=eps)
        adv_loader = DataLoader(dataset=TensorDataset(adv_x.cpu(), y_test),
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=1,
                                drop_last=False)
        _, adv_acc, _ = eval(model, criterion_cal, adv_loader)
        adv_accs.append(adv_acc)
        logging.info(f'test acc: {test_acc}, FGSM {eps} adv acc: {adv_acc}')

    return test_acc, adv_accs


def eval(model: nn.Module, criterion: nn.Module, data_loader: DataLoader):
    loss, correct = 0., 0
    labels, preds = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(args.device), y.to(args.device)
            out = model(x)
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            loss += criterion(out, y).item()
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(pred.tolist())
    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)
    bca = bca_score(labels, preds)

    return loss, acc, bca


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trades loss training')
    parser.add_argument('--gpu_id', type=str, default='3')
    parser.add_argument('--model', type=str, default='DeepCNN')
    parser.add_argument('--dataset', type=str, default='EPFL')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=5.0)
    parser.add_argument('--awp', type=bool, default=False)
    parser.add_argument('--awp_gamma', type=float, default=0.005)
    parser.add_argument('--setup', type=str, default='within')
    parser.add_argument('--log', type=str, default='')

    args = parser.parse_args()

    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    subject_num_dict = {'MI4C': 9, 'ERN': 16, 'EPFL': 8}

    model_path = f'model/trades/target/{args.dataset}/{args.model}/{args.setup}'

    npz_path = f'result/npz/trades/{args.setup}_{args.dataset}_{args.model}'
    if not os.path.exists(npz_path):
        os.makedirs(npz_path)
    npz_name = npz_path + '.npz'

    log_path = 'result/log/trades'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path,
                            f'{args.setup}_{args.dataset}_{args.model}.log')

    if len(args.log): log_name = log_name.replace('.log', f'_{args.log}')

    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # print logging
    print_log = logging.StreamHandler()
    logger.addHandler(print_log)
    # save logging
    save_log = logging.FileHandler(log_name, mode='w', encoding='utf8')
    logger.addHandler(save_log)

    logging.info(print_args(args) + '\n')

    r_acc, r_adv_acc = [], []
    for r in range(10):
        seed(r)
        # model train
        acc_list = []
        adv_acc_list = []
        for t in range(subject_num_dict[args.dataset]):
            # build model path
            model_save_path = os.path.join(model_path, f'{r}/{t}')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            logging.info(f'subject id: {t}')
            # load data
            if args.dataset == 'MI4C':
                x_train, y_train, x_test, y_test = MI4CLoad(id=t,
                                                            setup=args.setup)
            elif args.dataset == 'ERN':
                x_train, y_train, x_test, y_test = ERNLoad(id=t,
                                                           setup=args.setup)
            elif args.dataset == 'EPFL':
                x_train, y_train, x_test, y_test = EPFLLoad(id=t,
                                                            setup=args.setup)
            x_train, y_train, x_val, y_val = split(x_train,
                                                   y_train,
                                                   ratio=0.75)
            logging.info(f'train: {x_train.shape}, test: {x_test.shape}')
            x_train = Variable(
                torch.from_numpy(x_train).type(torch.FloatTensor))
            y_train = Variable(
                torch.from_numpy(y_train).type(torch.LongTensor))
            x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
            y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor))

            test_acc, adv_acc = train(x_train, y_train, x_test, y_test,
                                      model_save_path, args)
            acc_list.append(test_acc)
            adv_acc_list.append(adv_acc)

        r_acc.append(acc_list)
        r_adv_acc.append(adv_acc_list)
        np.savez(npz_path + f'/{r}.npz', acc=acc_list, adv_acc=adv_acc_list)

        logging.info(f'Repeat {r + 1}')
        logging.info(f'Mean acc: {np.mean(acc_list)}')
        logging.info(f'Mean adv acc: {np.mean(adv_acc_list, axis=0)}')
        logging.info(f'acc: {acc_list}')
        logging.info(f'adv acc: {adv_acc_list}')

    np.savez(npz_name, r_acc=r_acc, r_adv_acc=r_adv_acc)

    r_acc, r_adv_acc = np.mean(r_acc, axis=1), np.mean(r_adv_acc, axis=1)
    r_adv_acc = np.array(r_adv_acc)
    logging.info('*' * 50)
    logging.info(
        f'Repeat mean acc: {round(np.mean(r_acc), 4)}-{round(np.std(r_acc), 4)}'
    )
    for i in range(3):
        logging.info(
            f'Repeat mean adv acc (PGD {0.05 * (i + 1)}): {round(np.mean(r_adv_acc[:, i]), 4)}-{round(np.std(r_adv_acc[:, i]), 4)}'
        )
    for i in range(3, 6):
        logging.info(
            f'Repeat mean adv acc (FGSM {0.05 * (i - 2)}): {round(np.mean(r_adv_acc[:, i]), 4)}-{round(np.std(r_adv_acc[:, i]), 4)}'
        )
