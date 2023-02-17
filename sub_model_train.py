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
from torch.utils.data import TensorDataset, DataLoader
from model import EEGNet, DeepConvNet, ShallowConvNet
from utils.data_loader import MI4CLoad, ERNLoad, EPFLLoad, split
from utils.pytorch_utils import init_weights, print_args, seed, CustomTensorDataset
from pruning import pruning_by_ratio


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


def train_sub_model(x_sub: torch.Tensor, y_sub: torch.Tensor,
                    x_test: torch.Tensor, y_test: torch.Tensor,
                    aug_repeat: int, model_path: str, sub_model_path: str,
                    args):

    if args.defense == 'pruning':
        model = load_target_p_model(x_sub, y_sub, model_path, args)
    else:
        model = load_target_model(x_sub, y_sub, model_path, args)

    sub_model = EEGNet(n_classes=len(np.unique(y_sub.numpy())),
                       Chans=x_sub.shape[2],
                       Samples=x_sub.shape[3],
                       kernLenght=64,
                       F1=4,
                       D=2,
                       F2=8,
                       dropoutRate=0.25).to(args.device)
    sub_model.apply(init_weights)

    params = [v for _, v in sub_model.named_parameters()]
    optimizer = optim.Adam(params, lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # initial dataset
    model.eval()
    pred_sub = get_pred(model, x_sub, args)

    test_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                             batch_size=32,
                             shuffle=True,
                             num_workers=1,
                             drop_last=False)

    for r in range(aug_repeat):
        train_loader = DataLoader(dataset=TensorDataset(x_sub, pred_sub),
                            batch_size=32,
                            shuffle=True,
                            num_workers=1,
                            drop_last=False)
        for epoch in range(100):
            sub_model.train()
            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(args.device), batch_y.to(
                    args.device)
                optimizer.zero_grad()
                logit = sub_model(batch_x)
                loss = criterion(logit, batch_y)
                loss.backward()
                optimizer.step()
                sub_model.MaxNormConstraint()

            if (epoch + 1) % 10 == 0:
                sub_model.eval()
                train_loss, train_acc = eval(sub_model, criterion,
                                             train_loader)
                test_loss, test_acc = eval(sub_model, criterion, test_loader)

                logging.info(
                    'Repeat {}, Epoch {}/{}: train loss: {:.4f} train acc: {:.2f} | test loss: {:.4f} test acc: {:.2f}'
                    .format(r + 1, epoch + 1, 100, train_loss, train_acc,
                            test_loss, test_acc))

        # 样本 augment
        if r < aug_repeat - 1:
            adv_x = attack_lib.FGSM(sub_model, train_loader, eps=0.05)
            adv_y = get_pred(model, adv_x.cpu(), args)
            x_sub = torch.cat([x_sub, adv_x.cpu()], dim=0)
            pred_sub = torch.cat([pred_sub, adv_y.type(torch.LongTensor)], dim=0)
            del adv_x, adv_y

    torch.save(sub_model.state_dict(), sub_model_path + '/model.pt')

    # eval attack performance
    model.eval()
    _, test_acc = eval(model, criterion, test_loader)

    # attack using PGD and FGSM
    adv_accs = []
    for eps in [0.01, 0.05, 0.1]:
        adv_x = attack_lib.PGD(sub_model,
                               test_loader,
                               eps=eps,
                               alpha=eps / 10,
                               steps=20)
        adv_loader = DataLoader(dataset=TensorDataset(adv_x.cpu(), y_test),
                                batch_size=128,
                                shuffle=False,
                                num_workers=1,
                                drop_last=False)
        if args.transform:
            _, adv_acc = eval_transform(model, criterion, adv_x.cpu(), y_test)
        else:
            if args.defense == 'random_self_ensemble':
                _, adv_acc = ensemble_eval(model, criterion, adv_loader)
            else:
                _, adv_acc = eval(model, criterion, adv_loader)
        adv_accs.append(adv_acc)
        logging.info(f'test acc: {test_acc}, PGD {eps} adv acc: {adv_acc}')
    for eps in [0.01, 0.05, 0.1]:
        adv_x = attack_lib.FGSM(sub_model, test_loader, eps=eps)
        adv_loader = DataLoader(dataset=TensorDataset(adv_x.cpu(), y_test),
                                batch_size=128,
                                shuffle=False,
                                num_workers=1,
                                drop_last=False)
        if args.transform:
            _, adv_acc = eval_transform(model, criterion, adv_x.cpu(), y_test)
        else:
            if args.defense == 'random_self_ensemble':
                _, adv_acc = ensemble_eval(model, criterion, adv_loader)
            else:
                _, adv_acc = eval(model, criterion, adv_loader)
        adv_accs.append(adv_acc)
        logging.info(f'test acc: {test_acc}, FGSM {eps} adv acc: {adv_acc}')

    return test_acc, adv_accs


def get_pred(model: nn.Module, x: torch.Tensor, args):
    pred_x = torch.ones(size=(len(x), 1)).squeeze().type(torch.LongTensor)
    train_loader = DataLoader(dataset=TensorDataset(x, pred_x),
                            batch_size=32,
                            shuffle=False,
                            num_workers=1,
                            drop_last=False)

    idx = 0
    for batch_x, _ in train_loader:
        sub_pred = model(batch_x.to(args.device))
        sub_pred = nn.Softmax(dim=1)(sub_pred).cpu().argmax(dim=1)
        pred_x[idx:idx+len(sub_pred)] = sub_pred.type(torch.LongTensor)
        idx += len(sub_pred)
    
    return pred_x


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
    loss /= (20 * len(data_loader.dataset))
    acc = correct / len(data_loader.dataset)

    return loss, acc


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
    correct = probs.eq(y.cpu().view_as(probs)).sum().item()
    loss /= (len(x) * 10)
    acc = correct / len(x)

    return loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train')
    parser.add_argument('--gpu_id', type=str, default='4')
    parser.add_argument('--model', type=str, default='DeepCNN')
    parser.add_argument('--dataset', type=str, default='EPFL')
    parser.add_argument('--defense', type=str, default='random_self_ensemble')

    parser.add_argument('--setup', type=str, default='within')

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
        log_path = f'result_b_transfer/log/input_transform/{args.defense}'
        npz_path = f'result_b_transfer/npz/input_transform/{args.defense}/{args.setup}_{args.dataset}_{args.model}'
        model_path = f'model/input_transform/{args.defense}/target/{args.dataset}/{args.model}/{args.setup}'
        sub_model_path = f'model/input_transform/{args.defense}/sub/{args.dataset}/{args.model}/{args.setup}'
    else:
        log_path = f'result_b_transfer/log/{args.defense}'
        npz_path = f'result_b_transfer/npz/{args.defense}/{args.setup}_{args.dataset}_{args.model}'
        model_path = f'model/{args.defense}/target/{args.dataset}/{args.model}/{args.setup}'
        sub_model_path = f'model/{args.defense}/sub/{args.dataset}/{args.model}/{args.setup}'

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path,
                            f'{args.setup}_{args.dataset}_{args.model}.log')
    if not os.path.exists(npz_path):
        os.makedirs(npz_path)
    npz_name = npz_path + '.npz'

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
            # build model
            model_save_path = os.path.join(model_path, f'{r}/{t}')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            sub_model_save_path = os.path.join(sub_model_path, f'{r}/{t}')
            if not os.path.exists(sub_model_save_path):
                os.makedirs(sub_model_save_path)

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
            x_train = Variable(
                torch.from_numpy(x_train).type(torch.FloatTensor))
            y_train = Variable(
                torch.from_numpy(y_train).type(torch.LongTensor))
            x_val = Variable(torch.from_numpy(x_val).type(torch.FloatTensor))
            y_val = Variable(torch.from_numpy(y_val).type(torch.LongTensor))
            x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
            y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor))

            # train sub model
            test_acc, adv_acc = train_sub_model(
                x_val,
                y_val,
                x_test,
                y_test,
                aug_repeat=2,
                model_path=model_save_path,
                sub_model_path=sub_model_save_path,
                args=args)
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
