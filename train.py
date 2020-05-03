# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import PrototypicalLoss
from omniglot_dataset import OmniglotDataset
from miniImageNet_dataset import MiniImageNet
from protonet import ProtoNet, ProtoResNet
from parser_util import get_parser

from tqdm import tqdm
import numpy as np
import torch
import os

from collections import OrderedDict


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt, mode):

    if opt.dataset == 0:
        dataset = OmniglotDataset(mode=mode, root=opt.dataset_root)
        n_classes = len(np.unique(dataset.y))
        if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
            raise(Exception('There are not enough classes in the dataset in order ' +
                            'to satisfy the chosen classes_per_it. Decrease the ' +
                            'classes_per_it_{tr/val} option and try again.'))
    elif opt.dataset == 1:
        dataset = MiniImageNet(mode)
        n_classes = len(dataset.wnids)
        if mode == "train" and n_classes < opt.classes_per_it_tr:
            raise(Exception('There are not enough classes in the dataset in order ' +
                            'to satisfy the chosen classes_per_it. Decrease the ' +
                            'classes_per_it_{tr/val} option and try again.'))
        elif mode == "val" and n_classes < opt.classes_per_it_val:
            raise(Exception('There are not enough classes in the dataset in order ' +
                            'to satisfy the chosen classes_per_it. Decrease the ' +
                            'classes_per_it_{tr/val} option and try again.'))
        elif mode == "test" and n_classes < opt.classes_per_it_val:
            raise(Exception('There are not enough classes in the dataset in order ' +
                            'to satisfy the chosen classes_per_it. Decrease the ' +
                            'classes_per_it_{tr/val} option and try again.'))
    else:
        raise(Exception("No such dataset!!"))
    return dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader


def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if opt.dataset == 0:
        if opt.net == 1:
            model = ProtoNet().to(device)
        elif opt.net == 2:
            model = ProtoResNet().to(device)
    elif opt.dataset == 1:
        if opt.net == 1:
            model = ProtoNet(x_dim=3).to(device)
        elif opt.net == 2:
            model = ProtoResNet(x_dim=3).to(device)
    else:
        raise(Exception("No such dataset!!"))
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, loss_fn, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)

            weights = model.parameters()
            loss, acc = loss_fn(model_output, y, weights)

            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)

            weights = model.parameters()
            loss, acc = loss_fn(model_output, y, weights)

            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model, loss_fn):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)

            weights = model.parameters()
            _, acc = loss_fn(model_output, y, weights)

            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


def eval(opt):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    print("Evaluation mode")
    init_seed(options)
    test_dataloader = init_dataloader(options, 'test')

    model = init_protonet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))
    
    distance_fn = "cosine" if options.distance_fn==0 else "euclidean"
    test_loss_fn = PrototypicalLoss(options.num_support_val, distance_fn, options.regularizer)

    
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model,
         loss_fn=test_loss_fn)



def visual_data(opt, test_dataloader, model, loss_fn):
    '''
    Get data for visualization from the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    loss_dict = {}
    test_iter = iter(test_dataloader)
    for batch in test_iter:
        x, y = batch
        x, y = x.to(device), y.to(device)
        model_output = model(x)

        weights = model.parameters()
        loss, _ = loss_fn(model_output, y, weights)

        loss_dict[tuple(y)] = loss.item()
    ordered_loss_dict = OrderedDict(sorted(loss_dict.items(), key=lambda x: x[1]))


    return ordered_loss_dict


if __name__ == '__main__':
    options = get_parser().parse_args()
    print(options)

    # Evaluation mode
    eval(options)

    # Training mode
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    # dataset = init_dataset(options, 'test')
    # print(dataset[0])

    tr_dataloader = init_dataloader(options, 'train')
    val_dataloader = init_dataloader(options, 'val')
    test_dataloader = init_dataloader(options, 'test')

    model = init_protonet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)

    distance_fn = "cosine" if options.distance_fn==0 else "euclidean"

    train_loss_fn = PrototypicalLoss(options.num_support_tr, distance_fn, options.regularizer)
    test_loss_fn = PrototypicalLoss(options.num_support_val, distance_fn, options.regularizer)

    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                loss_fn=train_loss_fn,
                optim=optim,
                lr_scheduler=lr_scheduler)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res

    print('Testing with last model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model,
         loss_fn=test_loss_fn)
    
    model.load_state_dict(best_state)
    print('Testing with best model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model,
         loss_fn=test_loss_fn)

    model.load_state_dict(best_state)
    
    print("Generating the data for visualization..")
    ordered_loss_dict = visual_data(opt=options,
         test_dataloader=test_dataloader,
         model=model,
         loss_fn=test_loss_fn)

    torch.save(ordered_loss_dict, 'ordered_loss_dict.pt')
