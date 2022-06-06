import copy
import os
import logging
import torch
import argparse
import attack_lib
import defense_lib
import p_model
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import EEGNet, DeepConvNet, ShallowConvNet
from torch.utils.data import TensorDataset, DataLoader, dataset
from utils.data_loader import MI4CLoad, ERNLoad, EPFLLoad, split
from utils.pytorch_utils import init_weights, print_args, seed, weight_for_balanced_classes, bca_score, CustomTensorDataset
from pruning import pruning_by_ratio
from sklearn.metrics import accuracy_score


def load_target_model(x: torch.Tensor, y: torch.Tensor, model_path: str, args):
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
                             noise_std=args.noise_std,
                             SAP_frac=args.SAP_frac).to(args.device)
    elif args.model == 'DeepCNN':
        model = DeepConvNet(n_classes=len(np.unique(y.numpy())),
                                  Chans=x.shape[2],
                                  Samples=x.shape[3],
                                  dropoutRate=0.5,
                                  noise_std=args.noise_std,
                                  SAP_frac=args.SAP_frac).to(args.device)
    elif args.model == 'ShadowCNN':
        model = ShallowConvNet(n_classes=len(np.unique(y.numpy())),
                                     Chans=x.shape[2],
                                     Samples=x.shape[3],
                                     dropoutRate=0.5,
                                     noise_std=args.noise_std,
                                     SAP_frac=args.SAP_frac).to(args.device)
    else:
        raise 'No such model!'

    model.load_state_dict(
        torch.load(model_path + '/model.pt',
                   map_location=lambda storage, loc: storage))
    model.to(device=args.device)

    return model


def load_target_p_model(x: torch.Tensor, y: torch.Tensor, model_path: str,
                        args):
    # initialize the model
    if args.model == 'EEGNet':
        model = p_model.EEGNet(n_classes=len(np.unique(y.numpy())),
                               Chans=x.shape[2],
                               Samples=x.shape[3],
                               kernLenght=64,
                               F1=4,
                               D=2,
                               F2=8,
                               dropoutRate=0.25).to(args.device)
    elif args.model == 'DeepCNN':
        model = p_model.DeepConvNet(n_classes=len(np.unique(y.numpy())),
                                    Chans=x.shape[2],
                                    Samples=x.shape[3],
                                    dropoutRate=0.5).to(args.device)
    elif args.model == 'ShadowCNN':
        model = p_model.ShallowConvNet(n_classes=len(np.unique(y.numpy())),
                                       Chans=x.shape[2],
                                       Samples=x.shape[3],
                                       dropoutRate=0.5).to(args.device)
    else:
        raise 'No such model!'

    model.load_state_dict(
        torch.load(model_path + '/model.pt',
                   map_location=lambda storage, loc: storage))
    model.to(device=args.device)

    # pruning
    if args.defense == 'pruning':
        for p, v in model.named_parameters():
            if 'neuron_mask' in p:
                v = pruning_by_ratio(v, ratio=args.pruning_rate)

    return model


def evaluation(x_sub: torch.Tensor, y_sub: torch.Tensor, x_test: torch.Tensor,
               y_test: torch.Tensor, model_save_path: str, args):
    # initialize the model
    if args.defense == 'pruning':
        model = load_target_p_model(x_sub, y_sub, model_save_path, args)
    else:
        model = load_target_model(x_sub, y_sub, model_save_path, args)

    criterion = nn.CrossEntropyLoss().to(args.device)

    # data loader
    test_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False)

    model.eval()
    clean_preds, clean_labels = eval(model, criterion, test_loader)
    test_acc = accuracy_score(clean_labels, clean_preds)

    sub_model = attack_lib.TrainSub(model,
                                    x_sub=x_sub,
                                    y_sub=y_sub,
                                    aug_repeat=2)

    # attack
    if args.target: adv_label = torch.zeros_like(y_test)
    else: adv_label = y_test 
    eps_list = [0.01, 0.05, 0.1] if args.distance == 'inf' else [1.0, 1.5, 2.0]
    attack_list = ['PGD', 'FGSM', 'Sub', 'Sim']
    adv_accs = []
    for eps in eps_list:
        adv_acc_list = []
        for attack in attack_list:
            logging.info(f'eps: {eps}, attack: {attack}')
            if attack == 'PGD':
                adv_x = attack_lib.PGD(model,
                                       x_test,
                                       adv_label,
                                       eps=eps,
                                       alpha=eps / 10,
                                       steps=20,
                                       distance=args.distance,
                                       target=args.target)
            elif attack == 'FGSM':
                adv_x = attack_lib.FGSM(model,
                                        x_test,
                                        adv_label,
                                        eps=eps,
                                        distance=args.distance,
                                        target=args.target)
            elif attack == 'Sub':
                adv_x = attack_lib.PGD(sub_model,
                                       x_test,
                                       adv_label,
                                       eps=eps,
                                       alpha=eps / 10,
                                       steps=20,
                                       distance=args.distance,
                                       target=args.target)
            elif attack == 'Sim':
                adv_x, _ = attack_lib.SimBA(model,
                                         x_test,
                                         adv_label,
                                         max_iters=0.5,
                                         eps=eps,
                                         distance=args.distance,
                                         target=args.target)
            adv_loader = DataLoader(dataset=TensorDataset(adv_x.cpu(), y_test),
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    drop_last=False)
            if args.transform:
                preds, labels = eval_transform(model, criterion, adv_x.cpu(), y_test)
            else:
                if args.defense == 'random_self_ensemble':
                    preds, labels = ensemble_eval(model, criterion, adv_loader)
                else:
                    preds, labels = eval(model, criterion, adv_loader)
            if args.target:
                idx = np.logical_and((labels != 0), (clean_preds == clean_labels))
                adv_acc = accuracy_score(np.zeros_like(labels[idx]), preds[idx])
                logging.info(f'test acc: {test_acc}, {attack} {eps} adv asr: {adv_acc}')
            else:
                adv_acc = accuracy_score(labels, preds)
                logging.info(f'test acc: {test_acc}, {attack} {eps} adv acc: {adv_acc}')
            adv_acc_list.append(adv_acc)
        adv_accs.append(adv_acc_list)

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

    return np.array(preds), np.array(labels)


def ensemble_eval(model: nn.Module, criterion: nn.Module,
                  data_loader: DataLoader):
    loss, correct = 0., 0
    labels, preds = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(args.device), y.to(args.device)
            probs = torch.zeros(20, len(x), len(np.unique(y.cpu().numpy())))
            for repeat in range(20):
                out = model(x)
                pred = nn.Softmax(dim=1)(out).cpu()
                loss += criterion(out, y).item()
                probs[repeat, :, :] = pred
            probs = probs.mean(dim=0).argmax(dim=1)
            correct += probs.eq(y.cpu().view_as(probs)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(probs.tolist())

    return np.array(preds), np.array(labels)


def eval_transform(model: nn.Module, criterion: nn.Module, x: torch.Tensor,
                   y: torch.Tensor):
    transform = defense_lib.get_transform(transform_name=args.defense)
    loss, correct = 0., 0
    with torch.no_grad():
        probs = None
        for repeat in range(10):
            if probs is None:
                probs = torch.zeros(10, len(x), len(np.unique(y.numpy())))
            dataloader = DataLoader(dataset=CustomTensorDataset((x, y),
                                                                transform),
                                    batch_size=128,
                                    shuffle=False,
                                    num_workers=1,
                                    drop_last=False)
            preds = None
            for batch_x, batch_y in dataloader:
                # plot_raw(x[0].numpy().squeeze(), batch_x[0].numpy().squeeze(), file_name='fig/shift', is_norm=True)
                # exit()
                batch_x, batch_y = batch_x.to(args.device), batch_y.to(
                    args.device)
                out = model(batch_x)
                pred = nn.Softmax(dim=1)(out).cpu()
                loss += criterion(out, batch_y).item()

                preds = pred if preds is None else torch.cat(
                    (preds, pred), dim=0)

            probs[repeat, :, :] = preds
    probs = probs.mean(dim=0).argmax(dim=1)

    return probs.cpu().numpy(), y.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--model', type=str, default='EEGNet')
    parser.add_argument('--dataset', type=str, default='MI4C')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--setup', type=str, default='within')

    parser.add_argument('--defense', type=str, default='self_ensembel_AT')
    parser.add_argument('--distance', type=str, default='inf')
    parser.add_argument('--target', type=bool, default=True)

    args = parser.parse_args()

    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    args.noise_std = 0.25 if args.defense == 'random_self_ensemble' else None
    args.SAP_frac = 0.1 if args.defense == 'SAP' else None
    args.pruning_rate = 0.8 if args.defense == 'pruning' else None

    subject_num_dict = {'MI4C': 9, 'ERN': 16, 'EPFL': 8}

    args.transform = args.defense in [
        'guassian', 'sampling', 'shifting', 'shuffling', 'amchange', 'random'
    ]

    if args.transform:
        log_path = f'result_{args.distance}_new/log/input_transform/{args.defense}'
        npz_path = f'result_{args.distance}_new/npz/input_transform/{args.defense}/{args.setup}_{args.dataset}_{args.model}'
        model_path = f'model/input_transform/{args.defense}/target/{args.dataset}/{args.model}/{args.setup}'
    else:
        log_path = f'result_{args.distance}_new/log/{args.defense}'
        npz_path = f'result_{args.distance}_new/npz/{args.defense}/{args.setup}_{args.dataset}_{args.model}'
        model_path = f'model/{args.defense}/target/{args.dataset}/{args.model}/{args.setup}'

    
    if not os.path.exists(npz_path):
        os.makedirs(npz_path)
    npz_name = os.path.join(npz_path,
                            f'{args.setup}_{args.dataset}_{args.model}.npz')

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path,
                            f'{args.setup}_{args.dataset}_{args.model}.log')
    
    if args.target:
        npz_name = npz_name.replace('.npz', '_target.npz')
        log_name = log_name.replace('.log', '_target.log')

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
            # x_train, x_test = standard_normalize(x_train, x_test)
            logging.info(f'train: {x_train.shape}, test: {x_test.shape}')
            x_val = Variable(
                torch.from_numpy(x_val).type(torch.FloatTensor))
            y_val = Variable(
                torch.from_numpy(y_val).type(torch.LongTensor))
            x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
            y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor))

            test_acc, adv_acc = evaluation(x_val, y_val, x_test, y_test,
                                      model_save_path, args)
            acc_list.append(test_acc)
            adv_acc_list.append(adv_acc)

        r_acc.append(acc_list)
        r_adv_acc.append(adv_acc_list)

        logging.info(f'Repeat {r + 1}')
        logging.info(f'Mean acc: {np.mean(acc_list)}')
        logging.info(f'Mean adv acc: {np.mean(adv_acc_list, axis=0)}')
        logging.info(f'acc: {acc_list}')
        logging.info(f'adv acc: {adv_acc_list}')

    np.savez(npz_name, r_acc=r_acc, r_adv_acc=r_adv_acc)

    # [repeat, subject, eps, attack]
    r_acc, r_adv_acc = np.mean(r_acc, axis=1), np.mean(r_adv_acc, axis=1)
    r_adv_acc = np.array(r_adv_acc)
    attack_list = ['PGD', 'FGSM', 'Sub', 'Sim']
    eps_list = [0.01, 0.05, 0.1]
    logging.info(f'Repeat mean acc: {round(np.mean(r_acc), 4)}-{round(np.std(r_acc), 4)}')
    for i, eps in enumerate(eps_list):
        logging.info('*' * 25 + f' {eps} ' + '*' * 25)
        for j, attack in enumerate(attack_list):
            logging.info(f'{attack} acc:{round(np.mean(r_adv_acc[:, i, j]), 4)}-{round(np.std(r_adv_acc[:, i, j]), 4)}')