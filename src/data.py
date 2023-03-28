from copy import deepcopy
import numpy as np
import torch
from sklearn.datasets import make_blobs
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SplitMnistGenerator():
    def __init__(self, cl3=False, fashion=False, one_hot=False):

        self.cl3 = cl3
        self.one_hot = one_hot

        if fashion:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))])
            train_dataset = datasets.FashionMNIST('data', train=True, transform=transform, download=True)
            test_dataset = datasets.FashionMNIST('data', train=False, transform=transform, download=True)
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = datasets.MNIST('data', train=True, transform=transform, download=True)
            test_dataset = datasets.MNIST('data', train=False, transform=transform, download=True)

        train_data, val_data = torch.utils.data.random_split(train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_data, batch_size=len(train_data))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        val_loader = DataLoader(val_data, batch_size=len(val_data))

        self.X_train = next(iter(train_loader))[0].numpy().reshape(-1, 28 * 28)
        self.X_test = next(iter(test_loader))[0].numpy().reshape(-1, 28 * 28)
        self.X_val = next(iter(val_loader))[0].numpy().reshape(-1, 28 * 28)
        self.train_label = next(iter(train_loader))[1].numpy()
        self.test_label = next(iter(test_loader))[1].numpy()
        self.val_label = next(iter(val_loader))[1].numpy()

        print("X_train: {}".format(self.X_train.shape))#  (60000, 784)
        print("X_test: {}".format(self.X_test.shape)) # (10000, 784)
        print("X_val: {}".format(self.X_val.shape))  # (10000, 784)

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

        if self.cl3:
            self.y_dim = 10
        else:
            self.y_dim = 2

    def get_dims(self):
        return self.X_train.shape[1], self.y_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            if self.one_hot:
                next_y_train = np.zeros((next_x_train.shape[0], self.y_dim))
                if self.cl3:
                    next_y_train[:train_0_id.shape[0], self.cur_iter*2] = 1
                    next_y_train[train_0_id.shape[0]:, self.cur_iter*2 + 1] = 1
                else:
                    next_y_train[:train_0_id.shape[0], 0] = 1
                    next_y_train[train_0_id.shape[0]:, 1] = 1
            else:
                if self.cl3:
                    next_y_train = np.hstack((self.train_label[train_0_id], self.train_label[train_1_id]))
                else:
                    next_y_train = np.hstack(
                        (np.zeros((train_0_id.shape[0])), np.ones((train_1_id.shape[0])))
                    )

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            if self.one_hot:
                next_y_test = np.zeros((next_x_test.shape[0], self.y_dim))
                if self.cl3:
                    next_y_test[:test_0_id.shape[0], self.cur_iter*2] = 1
                    next_y_test[test_0_id.shape[0]:, self.cur_iter*2 + 1] = 1
                else:
                    next_y_test[:test_0_id.shape[0], 0] = 1
                    next_y_test[test_0_id.shape[0]:, 1] = 1
            else:
                if self.cl3:
                    next_y_test = np.hstack((self.test_label[test_0_id], self.test_label[test_1_id]))
                else:
                    next_y_test = np.hstack(
                        (np.zeros((test_0_id.shape[0])), np.ones((test_1_id.shape[0])))
                    )

            val_0_id = np.where(self.val_label == self.sets_0[self.cur_iter])[0]
            val_1_id = np.where(self.val_label == self.sets_1[self.cur_iter])[0]
            next_x_val = np.vstack((self.X_val[val_0_id], self.X_val[val_1_id]))

            if self.one_hot:
                next_y_val = np.zeros((next_x_val.shape[0], self.y_dim))
                if self.cl3:
                    next_y_val[:val_0_id.shape[0], self.cur_iter*2] = 1
                    next_y_val[val_0_id.shape[0]:, self.cur_iter*2 + 1] = 1
                else:
                    next_y_val[:val_0_id.shape[0], 0] = 1
                    next_y_val[val_0_id.shape[0]:, 1] = 1
            else:
                if self.cl3:
                    next_y_val = np.hstack((self.val_label[val_0_id], self.val_label[val_1_id]))
                else:
                    next_y_val = np.hstack(
                        (np.zeros((val_0_id.shape[0])), np.ones((val_1_id.shape[0])))
                    )

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test, next_x_val, next_y_val

class SplitCIFAR10Generator(SplitMnistGenerator):
    def __init__(self, val=False, cl3=False, data_aug=False):

        super(SplitCIFAR10Generator, self).__init__(cl3=cl3, fashion=False, one_hot=True)
        self.val = val
        self.cl3 = cl3

        # airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        # elif dataset == 'cifar100':
        #     mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        #     std = [x / 255 for x in [68.2, 65.4, 70.4]]


        if data_aug:
            train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
        else:
            train_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])

        trainset = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)
        train_data, val_data = torch.utils.data.random_split(trainset, [len(trainset) - 10000, 10000],
                                                             generator=torch.Generator().manual_seed(42))
        cifar10_testloader = DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=0)
        cifar10_trainloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True, num_workers=0)
        cifar10_valloader = DataLoader(val_data, batch_size=len(val_data), shuffle=True, num_workers=0)

        tmp = next(iter(cifar10_trainloader))
        self.X_train, self.train_label = tmp[0].numpy(), tmp[1].numpy()
        tmp = next(iter(cifar10_testloader))
        self.X_test, self.test_label = tmp[0].numpy(), tmp[1].numpy()
        tmp = next(iter(cifar10_valloader))
        self.X_val, self.val_label = tmp[0].numpy(), tmp[1].numpy()

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]

        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

        print("X_train: {}".format(self.X_train.shape))  # (60000, 784)
        print("X_test: {}".format(self.X_test.shape))  # (10000, 784)
        print("X_val: {}".format(self.X_val.shape))  # (10000, 784)

        if self.cl3:
            self.y_dim = 10
        else:
            self.y_dim = 2

    def get_dims(self):
        return self.X_train.shape[0], self.y_dim

class SplitCIFAR100Generator():
    def __init__(self, cl3=True, one_hot=True, max_iter=10, data_aug=False):

        # train, val, test (40000, 3072) (10000, 3072) (10000, 3072)
        self.cl3 = cl3
        self.one_hot = one_hot


        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]

        if data_aug:
            train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
        else:
            train_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])

        trainset = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)
        train_data, val_data = torch.utils.data.random_split(trainset, [len(trainset) - 10000, 10000],
                                                             generator=torch.Generator().manual_seed(42))
        cifar10_testloader = DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=0)
        cifar10_trainloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True, num_workers=0)
        cifar10_valloader = DataLoader(val_data, batch_size=len(val_data), shuffle=True, num_workers=0)

        cifar100_train = datasets.CIFAR100('data/',
                                           train=True, transform=transform, download=True)
        cifar100_test = datasets.CIFAR100('data/',
                                          train=False, transform=transform, download=True)
        cifar100_train, cifar100_val = torch.utils.data.random_split(cifar100_train, [len(cifar100_train) - 10000, 10000],
                                                             generator=torch.Generator().manual_seed(42))
        cifar100_testloader = DataLoader(cifar100_test, batch_size=len(cifar100_test), shuffle=False, num_workers=0)
        cifar100_trainloader = DataLoader(cifar100_train, batch_size=len(cifar100_train), shuffle=True, num_workers=0)
        cifar100_valloader = DataLoader(cifar100_val, batch_size=len(cifar100_val), shuffle=True, num_workers=0)

        cifar10_train = next(iter(cifar10_trainloader))
        cifar10_test  = next(iter(cifar10_testloader))
        cifar10_val   = next(iter(cifar10_valloader))
        cifar100_train = next(iter(cifar100_trainloader))
        cifar100_test  = next(iter(cifar100_testloader))
        cifar100_val  = next(iter(cifar100_valloader))
        self.X_train = np.concatenate([cifar10_train[0].numpy(), cifar100_train[0].numpy()], axis=0)
        self.X_test = np.concatenate([cifar10_test[0].numpy(), cifar100_test[0].numpy()], axis=0)
        self.X_val = np.concatenate([cifar10_val[0].numpy(), cifar100_val[0].numpy()], axis=0)
        self.train_label = np.concatenate([cifar10_train[1].numpy(), cifar100_train[1].numpy() + 10], axis=0)
        self.test_label = np.concatenate([cifar10_test[1].numpy(), cifar100_test[1].numpy() + 10], axis=0)
        self.val_label = np.concatenate([cifar10_val[1].numpy(), cifar100_val[1].numpy() + 10], axis=0)

        print("train sz: {}".format(self.X_train.shape))
        print("test sz: {}".format(self.X_test.shape))
        print("val sz: {}".format(self.X_val.shape))
        print("train label sz: {}".format(self.train_label.shape))
        print("test label sz: {}".format(self.test_label.shape))
        print("val label sz: {}".format(self.val_label.shape))

        self.class_sets = [
            list(range(0, 10)),
            list(range(10, 20)),
            list(range(20, 30)),
            list(range(30, 40)),
            list(range(40, 50)),
            list(range(50, 60)),
            list(range(60, 70)),
            list(range(70, 80)),
            list(range(80, 90)),
            list(range(90, 100)),
        ]

        self.classes = [i for i in range(100)]

        self.max_iter = max_iter #self.nr_classes / self.nr_classes_per_task
        self.cur_iter = 0
        self.nr_classes_per_task = 10
        if self.cl3:
            self.y_dim = self.nr_classes_per_task * self.max_iter

    def get_dims(self):
        # Get data input and output dimensions
        if self.cl3:
            return len(self.train_dataset) / self.nr_classes_per_task, self.nr_classes_per_task * self.max_iter
        else:
            return len(self.train_dataset) / self.nr_classes_per_task, self.nr_classes_per_task

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # training set
            idx = np.isin(
                self.train_label,
                self.classes[self.nr_classes_per_task * self.cur_iter:self.nr_classes_per_task * self.cur_iter + self.nr_classes_per_task]
            )
            next_x_train = self.X_train[idx]
            next_y_train = self.train_label[idx]
            if self.cl3:
                next_y_train = np.eye(self.max_iter * self.nr_classes_per_task)[next_y_train]

            # validation set
            idx = np.isin(
                self.val_label,
                self.classes[
                self.nr_classes_per_task * self.cur_iter:self.nr_classes_per_task * self.cur_iter + self.nr_classes_per_task]
            )
            next_x_val = self.X_val[idx]
            next_y_val = self.val_label[idx]
            if self.cl3:
                next_y_val = np.eye(self.max_iter * self.nr_classes_per_task)[next_y_val]

            # test set
            idx = np.isin(
                self.test_label,
                self.classes[
                self.nr_classes_per_task * self.cur_iter:self.nr_classes_per_task * self.cur_iter + self.nr_classes_per_task]
            )
            next_x_test = self.X_test[idx]
            next_y_test = self.test_label[idx]
            if self.cl3:
                next_y_test = np.eye(self.max_iter * self.nr_classes_per_task)[next_y_test]

            self.cur_iter += 1

        return next_x_train, next_y_train, next_x_test, next_y_test, next_x_val, next_y_val

    def reset(self):
        self.cur_iter = 0

class CIFAR100Generator():
    def __init__(self, cl3=True, one_hot=True):
        # train, val, test (40000, 3072) (10000, 3072) (10000, 3072)
        self.cl3 = cl3
        self.one_hot = one_hot

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        cifar100_train = datasets.CIFAR100('data/',
                                           train=True, transform=transform, download=True)
        cifar100_test = datasets.CIFAR100('data/',
                                          train=False, transform=transform, download=True)
        cifar100_train, cifar100_val = torch.utils.data.random_split(
            cifar100_train, [len(cifar100_train) - 10000, 10000],
            generator=torch.Generator().manual_seed(42),
        )
        cifar100_testloader = DataLoader(cifar100_test, batch_size=len(cifar100_test), shuffle=False, num_workers=0)
        cifar100_trainloader = DataLoader(cifar100_train, batch_size=len(cifar100_train), shuffle=True, num_workers=0)
        cifar100_valloader = DataLoader(cifar100_val, batch_size=len(cifar100_val), shuffle=True, num_workers=0)

        cifar100_train = next(iter(cifar100_trainloader))
        cifar100_test = next(iter(cifar100_testloader))
        cifar100_val = next(iter(cifar100_valloader))
        self.X_train = cifar100_train[0].numpy()
        self.X_test = cifar100_test[0].numpy()
        self.X_val = cifar100_val[0].numpy()
        self.train_label = cifar100_train[1].numpy()
        self.test_label = cifar100_test[1].numpy()
        self.val_label = cifar100_val[1].numpy()

        print("train sz: {}".format(self.X_train.shape))
        print("test sz: {}".format(self.X_test.shape))
        print("val sz: {}".format(self.X_val.shape))
        print("train label sz: {}".format(self.train_label.shape))
        print("test label sz: {}".format(self.test_label.shape))
        print("val label sz: {}".format(self.val_label.shape))

        self.classes = [i for i in range(100)]

        self.max_iter = 1  # self.nr_classes / self.nr_classes_per_task
        self.cur_iter = 0
        self.nr_classes_per_task = 100
        self.y_dim = 100


    def get_dims(self):
        # Get data input and output dimensions
        return len(self.train_dataset) / self.nr_classes_per_task, self.nr_classes_per_task

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            x_train = deepcopy(self.X_train)
            y_train = np.eye(self.y_dim)[self.train_label]

            # Retrieve test data
            x_test = deepcopy(self.X_test)
            y_test = np.eye(self.y_dim)[self.test_label]

            x_val = deepcopy(self.X_val)
            y_val = np.eye(self.y_dim)[self.val_label]
            self.cur_iter += 1
            return x_train, y_train, x_test, y_test, x_val, y_val

class PermutedMnistGenerator():
    def __init__(self, max_iter=10):

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST('data', train=False, transform=transform, download=True)

        train_data, val_data = torch.utils.data.random_split(train_dataset, [50000, 10000],
                                                             generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_data, batch_size=len(train_data))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        val_loader = DataLoader(val_data, batch_size=len(val_data))

        self.X_train = next(iter(train_loader))[0].numpy().reshape(-1, 28 * 28)
        self.X_test = next(iter(test_loader))[0].numpy().reshape(-1, 28 * 28)
        self.X_val = next(iter(val_loader))[0].numpy().reshape(-1, 28 * 28)
        self.train_label = next(iter(train_loader))[1].numpy()
        self.test_label = next(iter(test_loader))[1].numpy()
        self.val_label = next(iter(val_loader))[1].numpy()

        print("X_train: {}".format(self.X_train.shape))  # (60000, 784)
        print("X_test: {}".format(self.X_test.shape))  # (10000, 784)
        print("X_val: {}".format(self.X_val.shape))  # (10000, 784)
        print("train_label: {}".format(self.train_label.shape))  # (60000, 784)
        print("test_label: {}".format(self.test_label.shape))  # (10000, 784)
        print("val_label: {}".format(self.val_label.shape))  # (10000, 784)

        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter)
            perm_inds = list(range(self.X_train.shape[1]))
            np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]
            next_y_train = self.train_label

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:, perm_inds]
            next_y_test = self.test_label

            next_x_val = deepcopy(self.X_val)
            next_x_val = next_x_val[:, perm_inds]
            next_y_val = self.val_label

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test, next_x_val, next_y_val

class MnistGenerator():
    def __init__(self, fashion=False):

        if fashion:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))])
            train_dataset = datasets.FashionMNIST('data', train=True, transform=transform, download=True)
            test_dataset = datasets.FashionMNIST('data', train=False, transform=transform, download=True)
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = datasets.MNIST('data', train=True, transform=transform, download=True)
            test_dataset = datasets.MNIST('data', train=False, transform=transform, download=True)

        train_data, val_data = torch.utils.data.random_split(train_dataset, [50000, 10000],
                                                             generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_data, batch_size=len(train_data))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        val_loader = DataLoader(val_data, batch_size=len(val_data))

        self.X_train = next(iter(train_loader))[0].numpy().reshape(-1, 28 * 28)
        self.X_test = next(iter(test_loader))[0].numpy().reshape(-1, 28 * 28)
        self.X_val = next(iter(val_loader))[0].numpy().reshape(-1, 28 * 28)
        self.train_label = next(iter(train_loader))[1].numpy()
        self.test_label = next(iter(test_loader))[1].numpy()
        self.val_label = next(iter(val_loader))[1].numpy()

        print("X_train: {}".format(self.X_train.shape))  # (60000, 784)
        print("X_test: {}".format(self.X_test.shape))  # (10000, 784)
        print("X_val: {}".format(self.X_val.shape))  # (10000, 784)

        self.max_iter = 5
        self.y_dim = 10

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.y_dim

    def next_task(self):
        # Retrieve train data
        x_train = deepcopy(self.X_train)
        y_train = np.eye(10)[self.train_label]

        # Retrieve test data
        x_test = deepcopy(self.X_test)
        y_test = np.eye(10)[self.test_label]

        x_val = deepcopy(self.X_val)
        y_val = np.eye(10)[self.val_label]
        return x_train, y_train, x_test, y_test, x_val, y_val

class ToyGaussiansContGenerator():
    def __init__(self, max_iter=5, num_samples=2000, option=0, flatten_labels=True):

        self.offset = 5  # Offset when loading data in next_task()
        self.flatten_labels = flatten_labels
        # Generate data
        if option == 0:
            # Standard settings
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3],
                       [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16],
                   [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22]]

        elif option == 1:
            # Six tasks
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3], [1.65, 0.1],
                       [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1], [0.7, 0.25]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16], [0.14, 0.14],
                   [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22], [0.14, 0.14]]

        elif option == 2:
            # All std devs increased
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3],
                       [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1]]
            std = [[0.12, 0.22], [0.24, 0.12], [0.07, 0.2], [0.16, 0.08], [0.08, 0.16],
                   [0.12, 0.16], [0.16, 0.12], [0.08, 0.16], [0.24, 0.08], [0.08, 0.22]]

        elif option == 3:
            # Tougher to separate
            centers = [[0, 0.2], [0.6, 0.65], [1.3, 0.4], [1.6, -0.22], [2.0, 0.3],
                       [0.45, 0], [0.7, 0.55], [1., 0.1], [1.7, -0.3], [2.3, 0.1]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16],
                   [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22]]

        elif option == 4:
            # Two tasks, of same two gaussians
            centers = [[0, 0.2], [0, 0.2],
                       [0.45, 0], [0.45, 0]]
            std = [[0.08, 0.22], [0.08, 0.22],
                   [0.08, 0.16], [0.08, 0.16]]

        else:
            # If new / unknown option
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3],
                       [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16],
                   [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22]]

        if option != 1 and max_iter > 5:
            raise Exception("Current toydatagenerator only supports up to 5 tasks.")

        self.X, self.y = make_blobs(num_samples, centers=centers, cluster_std=std)
        self.X = self.X.astype('float32')
        self.X_test, self.y_test = make_blobs(num_samples, centers=centers, cluster_std=std)
        self.X_test = self.X_test.astype('float32')
        h = 0.01
        self.x_min, self.x_max = self.X[:, 0].min() - 0.2, self.X[:, 0].max() + 0.2
        self.y_min, self.y_max = self.X[:, 1].min() - 0.2, self.X[:, 1].max() + 0.2
        self.data_min = np.array([self.x_min, self.y_min], dtype='float32')
        self.data_max = np.array([self.x_max, self.y_max], dtype='float32')
        self.data_min = np.expand_dims(self.data_min, axis=0)
        self.data_max = np.expand_dims(self.data_max, axis=0)
        xx, yy = np.meshgrid(np.arange(self.x_min, self.x_max, h),
                             np.arange(self.y_min, self.y_max, h))
        xx = xx.astype('float32')
        yy = yy.astype('float32')
        self.test_shape = xx.shape
        X_test = np.c_[xx.ravel(), yy.ravel()]
        print(X_test.shape)
        self.X_test_full = X_test
        self.y_test_full = np.zeros(len(X_test), dtype=np.int8)
        print(self.y_test_full.shape)
        self.max_iter = max_iter
        self.num_samples = num_samples  # number of samples per task
        self.y_dim = 2

        if option == 1:
            self.offset = 6
        elif option == 4:
            self.offset = 2

        self.cur_iter = 0

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception("Number of tasks exceeded!")
        else:
            # train
            x_train_0 = self.X[self.y == self.cur_iter]
            x_train_1 = self.X[self.y == self.cur_iter + self.offset]
            y_train_0 = np.zeros_like(self.y[self.y == self.cur_iter])
            y_train_1 = np.ones_like(self.y[self.y == self.cur_iter + self.offset])
            x_train = np.concatenate([x_train_0, x_train_1], axis=0)
            y_train = np.concatenate([y_train_0, y_train_1], axis=0)
            y_train = np.eye(2)[y_train.reshape(-1)]
            y_train = y_train.astype('int64')
            # test
            x_test_0 = self.X_test[self.y_test == self.cur_iter]
            x_test_1 = self.X_test[self.y_test == self.cur_iter + self.offset]
            y_test_0 = np.zeros_like(self.y_test[self.y_test == self.cur_iter])
            y_test_1 = np.ones_like(self.y_test[self.y_test == self.cur_iter + self.offset])
            x_test = np.concatenate([x_test_0, x_test_1], axis=0)
            y_test = np.concatenate([y_test_0, y_test_1], axis=0)
            y_test = np.eye(2)[y_test.reshape(-1)]
            y_test = y_test.astype('int64')
            self.cur_iter += 1
            if self.flatten_labels:
                return x_train, y_train.argmax(1), x_test, y_test.argmax(1), self.X_test_full, self.y_test_full
            else:
                return x_train, y_train, x_test, y_test, self.X_test_full, self.y_test_full

    def full_data(self):
        x_train_list = []
        y_train_list = []
        for i in range(self.max_iter):
            x_train_list.append(self.X[self.y == i])
            x_train_list.append(self.X[self.y == i + self.offset])
            y_train_list.append(np.zeros_like(self.y[self.y == i]))
            y_train_list.append(np.ones_like(self.y[self.y == i + self.offset]))
        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_train = y_train.astype('int64')
        return x_train, y_train, self.X_test, self.y_test

    def reset(self):
        self.cur_iter = 0

    def get_dims(self):
        return 2, self.y_dim

class ToyGaussiansGenerator():
    def __init__(self, N, cl3, flatten_labels):
        self.N = N
        self.cur_iter = 0
        self.max_iter = 5
        self.y_dim = 10 if cl3 else 2
        self.cl3 = cl3
        self.flatten_labels = flatten_labels

    def get_dims(self):
        return 2, self.y_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')

        var = 0.02
        cov = np.array([[var, 0], [0, var]])
        x_train, x_test, x_val = [], [], []
        y_train, y_test, y_val = [], [], []
        for i in range(2):
            theta = 36 * (2 * self.cur_iter + i)
            r = 2
            rad = (theta * np.pi) / 180
            x = r * np.cos(rad)
            y = r * np.sin(rad)
            x_train.append(np.random.multivariate_normal(
                np.array([x, y]),
                cov,
                int(self.N),
            ))
            x_test.append(np.random.multivariate_normal(
                np.array([x, y]),
                cov,
                int(self.N / 4),
            ))
            x_val.append(np.random.multivariate_normal(
                np.array([x, y]),
                cov,
                int(self.N / 4),
            ))
            y_train.append(np.full(self.N, 2 * self.cur_iter + i if self.cl3 else i))
            y_test.append(np.full(int(self.N / 4), 2 * self.cur_iter + i if self.cl3 else i))
            y_val.append(np.full(int(self.N / 4), 2 * self.cur_iter + i if self.cl3 else i))

        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train).reshape(-1, 1)
        n, d = self.get_dims()
        y_train_1d = np.copy(y_train)
        y_train = np.zeros((y_train_1d.shape[0], d))
        if self.cl3:
            y_train = np.eye(10)[y_train_1d.reshape(-1)]
        else:
            y_train[:, 0] = y_train_1d.reshape(-1)
            y_train[:, 1] = 1 - y_train_1d.reshape(-1)
        perm_inds = list(range(x_train.shape[0]))
        np.random.shuffle(perm_inds)
        x_train = x_train[perm_inds, :]
        y_train = y_train[perm_inds, :]

        x_test = np.concatenate(x_test)
        y_test = np.concatenate(y_test)
        y_test_1d = np.copy(y_test)
        y_test = np.zeros((y_test_1d.shape[0], d))
        if self.cl3:
            y_test = np.eye(10)[y_test_1d.reshape(-1)]
        else:
            y_test[:, 0] = y_test_1d.reshape(-1)
            y_test[:, 1] = 1 - y_test_1d.reshape(-1)
        perm_inds = list(range(x_test.shape[0]))
        np.random.shuffle(perm_inds)
        x_test = x_test[perm_inds, :]
        y_test = y_test[perm_inds, :]

        x_val = np.concatenate(x_val)
        y_val = np.concatenate(y_val)
        y_val_1d = np.copy(y_val)
        y_val = np.zeros((y_val_1d.shape[0], d))
        if self.cl3:
            y_val = np.eye(10)[y_val_1d.reshape(-1)]
        else:
            y_val[:, 0] = y_val_1d.reshape(-1)
            y_val[:, 1] = 1 - y_val_1d.reshape(-1)
        perm_inds = list(range(x_val.shape[0]))
        np.random.shuffle(perm_inds)
        x_val = x_val[perm_inds, :]
        y_val = y_val[perm_inds, :]

        self.cur_iter += 1

        if self.flatten_labels:
            return x_train, y_train.argmax(1), x_test, y_test.argmax(1), x_val, y_val.argmax(1)
        else:
            return x_train, y_train, x_test, y_test, x_val, y_val
