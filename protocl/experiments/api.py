import os
import sys
sys.path.append('../../')
import datetime
import copy
import pickle
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, TensorDataset, DataLoader

from protocl.main.utils import mask_nlls, compute_acc, scores_to_arr, Storage


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def to_cuda(x, config):
    if config['data.cuda'] >= 0:
        return x.float().cuda(config['data.cuda'])
    else:
        return x.float()

def test(
    test_loaders,
    model,
    config,
    task_id,
    n_tasks, 
    results,
    seed,
    all_acc,
    all_nll,
):
    accs_lst, nlls_lst = [], []
    true_labels, pred_labels = [], []

    for test_task_id, test_loader in enumerate(test_loaders):

        print("Test task: {}".format(test_task_id))

        with torch.no_grad():

            np.random.seed(seed)
            torch.manual_seed(seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            running_acc, running_nll = [], []

            for i, (data, target) in enumerate(test_loader):

                if data.shape[0] == config['data.batch_size']:

                    inputs = to_cuda(data, config)
                    labels = to_cuda(target, config)

                    _, _nlls = model.log_posterior(inputs, labels)

                    # nlls are shape batchsize x n_classes, so we need to mask
                    # for loss and also eval accuracy
                    accs = compute_acc(labels, _nlls)  # batchsize x 1
                    nlls = mask_nlls(labels, _nlls)  # now batchsize x 1

                    mean_nll = torch.mean(nlls)

                    running_nll.append(mean_nll.item())
                    mean_acc = torch.mean(accs)
                    running_acc.append(mean_acc.item())

            # compute stats on NLL and accuracy

            mean_nll = np.mean(running_nll)
            nll_stderr = np.std(running_nll) / np.sqrt(len(running_nll))
            nlls_lst.append(mean_nll)

            print('Test Mean NLL: ', mean_nll)
            print('Test NLL 95% confidence: +/-', nll_stderr * 1.96)

            results[task_id][test_task_id] = {
                'running_nll': running_nll,
                'mean_nll': mean_nll,
                'nll_conf': nll_stderr * 1.96,
            }

            mean_acc = np.mean(running_acc)
            acc_stderr = np.std(running_acc) / np.sqrt(len(running_acc))
            accs_lst.append(mean_acc)

            print('Mean accuracy: ', 100. * mean_acc)
            print('Accuracy 95% confidence: +/-', acc_stderr * 196.)

            results[task_id][test_task_id]['running_acc'] = running_acc
            results[task_id][test_task_id]['mean_acc'] = 100. * mean_acc
            results[task_id][test_task_id]['acc_conf'] = 196. * acc_stderr

            print('---------------\n')

            # get a confusion matrix of true class versus predicted class
            # for the last task only
            if task_id == n_tasks - 1:
                pred_class = torch.argmin(_nlls, -1, keepdim=True).data.cpu().numpy()
                pred_labels.append(pred_class)
                true_labels.append(labels.data.cpu().numpy())

    all_acc = scores_to_arr(all_acc, accs_lst)
    all_nll = scores_to_arr(all_nll, nlls_lst)
    return all_acc, all_nll, results, true_labels, pred_labels

def run(
    model,
    config,
    n_tasks,
    data_gen,
    n_epochs,
    batch_size,
    path,
    seed,
):

    print("Initialising model")
    model = to_cuda(model, config)

    running_nll, running_acc = [], []
    test_loaders = []
    task_memory = {}

    all_acc, all_nll = np.array([]), np.array([])
    results = {i: {i: {} for i in range(n_tasks)} for i in range(n_tasks)}

    print("CL training loop")
    for task_id in range(n_tasks):

        print("\n Task {} \n".format(task_id + 1))

        writer = SummaryWriter(os.path.join(DIR_PATH, './runs/' + path + datetime.datetime.now().strftime('y%y_m%m_d%d_s%s') + '_task' + str(task_id + 1)))

        x_train, y_train, x_test, y_test, x_val, y_val = data_gen.next_task()

        print("train set: {}".format(x_train.shape))

        train_loader = DataLoader(
            TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train).long()),
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
        )
        test_loader = DataLoader(
            TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test).long()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val).long()),
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
        )

        test_loaders.append(copy.deepcopy(val_loader) if config['train.val'] else copy.deepcopy(test_loader))

        # Augment task dataset with memory
        # Code from https://github.com/GT-RIPL/Continual-Learning-Benchmark/
        # @inproceedings{Hsu18_EvalCL,
        #   title={Re-evaluating Continual Learning Scenarios: A Categorization and Case for Strong Baselines},
        #   author={Yen-Chang Hsu and Yen-Cheng Liu and Anita Ramasamy and Zsolt Kira},
        #   booktitle={NeurIPS Continual learning Workshop },
        #   year={2018},
        #   url={https://arxiv.org/abs/1810.12488},
        # }
        if config['memory.memory_size'] == 0:
            new_train_loader = train_loader
        else:
            dataset_list = []
            for storage in task_memory.values():
                dataset_list.append(storage)
            dataset_list *= max(len(train_loader.dataset) // config['memory.memory_size'], 1)  # Let old data: new data = 1:1
            dataset_list.append(train_loader.dataset)
            dataset = torch.utils.data.ConcatDataset(dataset_list)
            new_train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers
            )

        iter = 0
        for epoch in range(n_epochs):
            for i, (data, target) in enumerate(new_train_loader):
                iter = epoch * len(new_train_loader) + i
                if data.shape[0] == config['data.batch_size']:

                    if (i + 1) % config['train.decay_every'] == 0:
                        model.scheduler.step()
                        if config['train.verbose']:
                            print('decreasing learning rate\n')

                    inputs = to_cuda(data, config)
                    labels = to_cuda(target, config)

                    params, nlls = model.log_posterior(inputs, labels)
                    # nlls are shape batchsize x n_classes, so we need to mask
                    # for loss and also eval accuracy
                    accs = compute_acc(labels, nlls)  # batchsize x 1
                    nlls = mask_nlls(labels, nlls)  # now batchsize x 1

                    mean_nll = torch.mean(nlls)
                    mean_nll.backward()
                    model.optimizer.step()

                    # TensorBoard logging
                    running_nll.append(mean_nll.item())
                    writer.add_scalar(
                        'NLL/Train',
                        mean_nll.item(), 
                        iter,
                    )

                    ep_acc = torch.mean(accs).item()
                    writer.add_scalar(
                        'Accuracy/Train', 
                        ep_acc, 
                        iter,
                    )
                    writer.add_scalar(
                        'LR',
                        model.scheduler.get_last_lr()[0], 
                        iter,
                    )
                    running_acc.append(ep_acc)
                    writer.add_scalar(
                        'max_Linv_eigenval',
                        np.mean(np.exp(model.logLinv.data.cpu().numpy()).max(0)),
                        iter,
                    )
                    writer.add_scalar(
                        'min_Linv_eigenval',
                        np.mean(np.exp(model.logLinv.data.cpu().numpy()).min(0)), 
                        iter,
                    )

                    writer.add_scalar(
                        'max_q_val', 
                        np.mean(model.Q.data.cpu().numpy().max(0)), 
                        iter,
                    )
                    writer.add_scalar(
                        'min_q_val', 
                        np.mean(model.Q.data.cpu().numpy().min(0)), 
                        iter,
                    )

                    dirs_prior = model.log_dirichlet_priors.data.cpu().numpy()
                    dirs_post = np.mean(params[2].cpu().numpy(), 0)
                    for k in range(data_gen.y_dim):
                        writer.add_scalar(
                            'dir_weight_prior_{}'.format(k+1),
                            np.exp(np.mean(dirs_prior, 0)[k]) / np.sum(np.exp(np.mean(dirs_prior, 0))), 
                            iter,
                        )
                        writer.add_scalar(
                            'dir_weight_post_{}'.format(k+1),
                            dirs_post[k] / np.sum(dirs_post),
                            iter,
                        )
                    
                    log_sig_eps = model.logSigEps.data.cpu().numpy()
                    if config['train.learnable_noise']:
                        writer.add_scalar(
                            'max_sigeps_eigenval',
                            np.mean(np.exp(log_sig_eps).max(0)),
                            iter,
                        )
                        writer.add_scalar(
                            'min_sigeps_eigenval', 
                            np.mean(np.exp(log_sig_eps).min(0)),
                            iter,
                        )

                    # Validation
                    if iter % 1000 == 0:

                        print('Iteration ' + str(iter) + '/' + str(len(new_train_loader) * n_epochs))
                        print('Train NLL: ', np.mean(running_nll))
                        running_nll = []

                        train_acc = np.mean(running_acc)
                        print('Train Accuracy: ', train_acc)
                        running_acc = []

                        if config['train.val_iterations'] != 0:
                            with torch.no_grad():
                                running_val_nll = []
                                running_val_acc = []
                                for j, (data, target) in enumerate(val_loader):

                                    val_inputs = to_cuda(data, config)
                                    val_labels = to_cuda(target, config)

                                    _, val_nlls = model.log_posterior(val_inputs, val_labels)

                                    # nlls are shape batchsize x n_classes, so we need to mask
                                    # for loss and also eval accuracy
                                    val_accs = compute_acc(val_labels, val_nlls)  # batchsize x 1
                                    val_nlls = mask_nlls(val_labels, val_nlls)  # now batchsize x 1

                                    val_mean_nll = torch.mean(val_nlls)
                                    running_val_nll.append(val_mean_nll.item())

                                    running_val_acc.append(torch.mean(val_accs).item())

                                    if j == config['train.val_iterations']:
                                        val_nll = np.mean(running_val_nll)

                                        writer.add_scalar('NLL/Validation', val_nll, iter)
                                        val_acc = torch.mean(val_accs)
                                        writer.add_scalar('Accuracy/Validation', val_acc.item(), iter)

                                        # Log gradients
                                        if config['train.debug_grads']:
                                            for n, p in model.named_parameters():
                                                if p.grad is not None:
                                                    writer.add_histogram('grad_{}'.format(n), p.grad, iter)

                                        break

                                if config['train.verbose']:
                                    print('Validation NLL: ', np.mean(running_val_nll))
                                    print('Validation Accuracy: ', val_acc.item())

                        print('--------------------')

                    model.optimizer.zero_grad()

        # Randomly decide which images/points to keep in the dataset
        # (a) Decide the number of samples for being saved
        num_sample_per_task = config['memory.memory_size'] // (task_id + 1)
        num_sample_per_task = min(len(train_loader.dataset), num_sample_per_task)
        # (b) Reduce current exemplar set to reserve the space for the new dataset
        for storage in task_memory.values():
            storage.reduce(num_sample_per_task)
        # (c) Randomly choose some samples from new task and save them to the memory
        randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        task_memory[(task_id + 1)] = Storage(train_loader.dataset, randind)

        all_acc, all_nll, results, true_labels, pred_labels = test(
            test_loaders,
            model,
            config,
            task_id,
            n_tasks,
            results,
            seed,
            all_acc,
            all_nll,
        )

    # Cache results for plotting etc.
    results['all_acc'] = all_acc
    results['all_nll'] = all_nll
    print("All accs: {}".format(all_acc))
    filename = 'results/' + path + '_final_res.pickle'
    with open(os.path.join(DIR_PATH, filename), 'wb') as f:
        pickle.dump(
            {
                'res': results,
                'confusion_matrix': {
                    'true_labels': true_labels,
                    'pred_labels': pred_labels,
                    },
                'config': config,
            },
            f,
        )