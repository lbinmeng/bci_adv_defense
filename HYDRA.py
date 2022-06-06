import copy
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
from torch.utils.data import TensorDataset, DataLoader, dataset
from p_model import EEGNet, DeepConvNet, ShallowConvNet
from utils.data_loader import MI4CLoad, ERNLoad, EPFLLoad, split
from utils.pytorch_utils import init_weights, init_mask, print_args, seed, weight_for_balanced_classes
from utils.plot_utils import plot_mask_dist


def train_mask(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               x_test: torch.Tensor, y_test: torch.Tensor, load_path: str,
               save_path: str, args):
    # pruning model
    # loading pre-train model
    pretrained_dict = torch.load(load_path + '/model.pt',
                                 map_location=lambda storage, loc: storage)
    model_dict = model.state_dict()
    predtrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in model_dict
    }
    model_dict.update(predtrained_dict)
    model.load_state_dict(model_dict)
    # model.apply(init_mask)
    model.to(device=args.device)

    mask_params = [
        v for p, v in model.named_parameters() if 'neuron_mask' in p
    ]
    mask_optim = optim.Adam(mask_params, lr=args.lr, weight_decay=5e-4)

    criterion_cal = nn.CrossEntropyLoss().to(args.device)
    criterion_kl = nn.KLDivLoss(reduction='mean').to(args.device)

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
                             shuffle=True,
                             num_workers=1,
                             drop_last=False)

    # optimize the mask
    model.eval()
    for epoch in range(150):
        adjust_learning_rate(optimizer=mask_optim, epoch=epoch + 1, args=args)
        l = 0.0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            batch_adv_x = attack_lib.PGD_batch(model,
                                               batch_x,
                                               batch_y,
                                               eps=0.05,
                                               alpha=0.005,
                                               steps=10)

            mask_optim.zero_grad()

            adv_logits = model(batch_adv_x)
            logits = model(batch_x)

            loss_cal = criterion_cal(logits, batch_y)
            loss_rob = criterion_kl(F.log_softmax(adv_logits, dim=1),
                                    F.softmax(logits, dim=1))
            loss = loss_cal + args.beta * loss_rob

            loss.backward()
            mask_optim.step()
            clip_mask(model)
            l += loss.item()

        if (epoch + 1) % 10 == 0:
            train_loss, train_acc = eval(model, criterion_cal, train_loader)
            test_loss, test_acc = eval(model, criterion_cal, test_loader)

            logging.info(
                'Epoch {}/{}: train loss: {:.4f} train acc: {:.2f} | test loss: {:.4f} test acc: {:.2f}'
                .format(epoch + 1, args.epochs, train_loss, train_acc,
                        test_loss, test_acc))
    plot_mask_dist(mask_params)
    torch.save(model.state_dict(), save_path + '/model.pt')

    _, test_acc = eval(model, criterion_cal, test_loader)

    # attack using PGD and FGSM
    adv_accs = []
    for eps in [0.01, 0.05, 0.1]:
        adv_x = attack_lib.PGD(model,
                               test_loader,
                               eps=eps,
                               alpha=eps / 10,
                               steps=20)
        adv_loader = DataLoader(dataset=TensorDataset(adv_x.cpu(), y_test),
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=1,
                                drop_last=False)
        _, adv_acc = eval(model, criterion_cal, adv_loader)
        adv_accs.append(adv_acc)
        logging.info(f'test acc: {test_acc}, PGD {eps} adv acc: {adv_acc}')
    for eps in [0.01, 0.05, 0.1]:
        adv_x = attack_lib.FGSM(model, test_loader, eps=eps)
        adv_loader = DataLoader(dataset=TensorDataset(adv_x.cpu(), y_test),
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=1,
                                drop_last=False)
        _, adv_acc = eval(model, criterion_cal, adv_loader)
        adv_accs.append(adv_acc)
        logging.info(f'test acc: {test_acc}, FGSM {eps} adv acc: {adv_acc}')

    return test_acc, adv_accs


def pruning(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
            x_test: torch.Tensor, y_test: torch.Tensor, load_path: str,
            save_path: str, args):
    # pruning model
    # loading masked model
    model.load_state_dict(
        torch.load(load_path + '/model.pt',
                   map_location=lambda storage, loc: storage))
    model.to(device=args.device)

    # pruning
    for p, v in model.named_parameters():
        if 'neuron_mask' in p:
            v = pruning_by_ratio(v, ratio=args.pruning_rate)

    # finetuning
    params = [v for p, v in model.named_parameters() if 'neuron_mask' not in p]
    param_optim = optim.Adam(params, lr=args.lr, weight_decay=1e-4)

    criterion_cal = nn.CrossEntropyLoss().to(args.device)
    criterion_kl = nn.KLDivLoss(reduction='mean').to(args.device)

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
                             shuffle=True,
                             num_workers=1,
                             drop_last=False)

    # finetuning
    for epoch in range(50):
        adjust_learning_rate(optimizer=param_optim, epoch=epoch + 1, args=args)
        model.train()
        l = 0.0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            batch_adv_x = attack_lib.PGD_batch(model,
                                               batch_x,
                                               batch_y,
                                               eps=0.05,
                                               alpha=0.005,
                                               steps=10)

            param_optim.zero_grad()

            adv_logits = model(batch_adv_x)
            logits = model(batch_x)

            loss_cal = criterion_cal(logits, batch_y)
            loss_rob = criterion_kl(F.log_softmax(adv_logits, dim=1),
                                    F.softmax(logits, dim=1))
            loss = loss_cal + args.beta * loss_rob

            loss.backward()
            param_optim.step()
            l += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            train_loss, train_acc = eval(model, criterion_cal, train_loader)
            test_loss, test_acc = eval(model, criterion_cal, test_loader)

            logging.info(
                'Epoch {}/{}: train loss: {:.4f} train acc: {:.2f} | test loss: {:.4f} test acc: {:.2f}'
                .format(epoch + 1, args.epochs, train_loss, train_acc,
                        test_loss, test_acc))

    torch.save(model.state_dict(), save_path + '/model.pt')
    model.eval()
    _, test_acc = eval(model, criterion_cal, test_loader)

    # attack using PGD and FGSM
    adv_accs = []
    for eps in [0.01, 0.05, 0.1]:
        adv_x = attack_lib.PGD(model,
                               x_test,
                               y_test,
                               eps=eps,
                               alpha=eps / 10,
                               steps=20)
        adv_loader = DataLoader(dataset=TensorDataset(adv_x.cpu(), y_test),
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=1,
                                drop_last=False)
        _, adv_acc = eval(model, criterion_cal, adv_loader)
        adv_accs.append(adv_acc)
        logging.info(f'test acc: {test_acc}, PGD {eps} adv acc: {adv_acc}')
    for eps in [0.01, 0.05, 0.1]:
        adv_x = attack_lib.FGSM(model, x_test, y_test, eps=eps)
        adv_loader = DataLoader(dataset=TensorDataset(adv_x.cpu(), y_test),
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=1,
                                drop_last=False)
        _, adv_acc = eval(model, criterion_cal, adv_loader)
        adv_accs.append(adv_acc)
        logging.info(f'test acc: {test_acc}, FGSM {eps} adv acc: {adv_acc}')

    return test_acc, adv_accs


def pruning_by_ratio(mask, ratio):
    m = mask.clone()
    _, idx = m.flatten().sort()
    th = int((1 - ratio) * m.numel())

    with torch.no_grad():
        temp = mask.flatten()
        temp[idx[:th]] = 0
        temp[idx[th:]] = 1.0
    return mask


def clip_mask(model, lower=0.0, upper=1.0):
    mask_params = [
        v for p, v in model.named_parameters() if 'neuron_mask' in p
    ]
    with torch.no_grad():
        for param in mask_params:
            param.clamp_(lower, upper)


def eval(model: nn.Module, criterion: nn.Module, data_loader: DataLoader):
    loss, correct = 0., 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(args.device), y.to(args.device)
            out = model(x)
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            loss += criterion(out, y).item()
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return loss, acc


def adjust_learning_rate(optimizer: nn.Module, epoch: int, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 50:
        lr = args.lr * 0.1
    if epoch >= 100:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pruning')
    parser.add_argument('--gpu_id', type=str, default='5')
    parser.add_argument('--model', type=str, default='ShadowCNN')
    parser.add_argument('--dataset', type=str, default='EPFL')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--setup', type=str, default='within')
    parser.add_argument('--beta', type=float, default=5.0)
    parser.add_argument('--phase', type=str, default='mask')
    parser.add_argument('--pruning_rate', type=float, default=0.8)

    args = parser.parse_args()

    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    subject_num_dict = {'MI4C': 9, 'ERN': 16, 'EPFL': 8}

    pretrain_model_path = f'model/pruning/origin/{args.dataset}/{args.model}/{args.setup}'
    mask_model_path = f'model/pruning/mask/{args.dataset}/{args.model}/{args.setup}'
    model_path = f'model/pruning/target/{args.dataset}/{args.model}/{args.setup}'

    npz_path = f'result/npz/pruning/{args.setup}_{args.dataset}_{args.model}'
    if not os.path.exists(npz_path):
        os.makedirs(npz_path)
    npz_name = npz_path + '.npz'

    log_path = 'result/log/pruning'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path,
                            f'{args.setup}_{args.dataset}_{args.model}.log')

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

    # model train
    r_acc, r_adv_acc = [], []
    for r in range(10):
        seed(r)
        # model train
        acc_list = []
        adv_acc_list = []
        for t in range(subject_num_dict[args.dataset]):
            # build model path
            pretrain_model_save_path = os.path.join(pretrain_model_path,
                                                    f'{r}/{t}')
            if not os.path.exists(pretrain_model_save_path):
                os.makedirs(pretrain_model_save_path)
            mask_save_path = os.path.join(mask_model_path, f'{r}/{t}')
            if not os.path.exists(mask_save_path):
                os.makedirs(mask_save_path)
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
            _, _, x_val, y_val = split(x_train,
                                       y_train,
                                       ratio=0.75,
                                       shuffle=False)
            # x_train, x_test = standard_normalize(x_train, x_test)
            logging.info(f'train: {x_train.shape}, test: {x_test.shape}')
            x_train = Variable(
                torch.from_numpy(x_train).type(torch.FloatTensor))
            y_train = Variable(
                torch.from_numpy(y_train).type(torch.LongTensor))
            x_val = Variable(torch.from_numpy(x_val).type(torch.FloatTensor))
            y_val = Variable(torch.from_numpy(y_val).type(torch.LongTensor))
            x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
            y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor))

            if args.model == 'EEGNet':
                model = EEGNet(n_classes=len(np.unique(y_train.numpy())),
                               Chans=x_train.shape[2],
                               Samples=x_train.shape[3],
                               kernLenght=64,
                               F1=4,
                               D=2,
                               F2=8,
                               dropoutRate=0.25,
                               norm_rate=0.25).to(args.device)
            elif args.model == 'DeepCNN':
                model = DeepConvNet(n_classes=len(np.unique(y_train.numpy())),
                                    Chans=x_train.shape[2],
                                    Samples=x_train.shape[3],
                                    dropoutRate=0.5).to(args.device)
            elif args.model == 'ShadowCNN':
                model = ShallowConvNet(n_classes=len(np.unique(
                    y_train.numpy())),
                                       Chans=x_train.shape[2],
                                       Samples=x_train.shape[3],
                                       dropoutRate=0.5).to(args.device)
            else:
                raise 'No such model!'

            if args.phase == 'mask':
                test_acc, adv_acc = train_mask(model, x_val, y_val, x_test,
                                               y_test,
                                               pretrain_model_save_path,
                                               mask_save_path, args)
            elif args.phase == 'pruning':
                test_acc, adv_acc = pruning(model, x_val, y_val, x_test,
                                            y_test, mask_save_path,
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