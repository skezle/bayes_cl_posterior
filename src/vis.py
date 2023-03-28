import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn.functional as F
import hamiltorch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_2d(model, params_hmc, tau_list, prior, test_full_loader, datagen, task_idx, tag):
    # Plot visualisation (2D figure)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_full_loader):
            if batch_idx >= 1:
                break

            x_test, y_test = data.to(device), target.to(device)
            pred_list, log_prob_list = hamiltorch.predict_model(
                model=model,
                x=x_test,
                y=y_test,
                samples=params_hmc,
                model_loss='multi_class_log_softmax_output',
                tau_out=1.,
                tau_list=tau_list,
                prior=prior,
            )
            ensemble_proba = F.softmax(pred_list.mean(0), dim=-1)
            _, pred = torch.max(pred_list.mean(0), -1)
            acc = (pred.float() == y_test.flatten()).sum().float() / y_test.shape[0]
            print("acc: {}".format(acc))

    print(ensemble_proba.shape)
    pred_y_show = ensemble_proba[:, 0] - ensemble_proba[:, 1]

    pred_y_show = pred_y_show.detach()
    pred_y_show = pred_y_show.cpu().numpy()
    cl_show = pred_y_show.reshape(datagen.test_shape)
    color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    plt.figure()
    axs = plt.subplot(111)
    axs.title.set_text('HMC SH')
    im = plt.imshow(cl_show, cmap='Blues',
                    extent=(datagen.x_min, datagen.x_max, datagen.y_min, datagen.y_max), origin='lower')
    for l in range(task_idx + 1):
        idx = np.where(datagen.y == l)
        plt.scatter(datagen.X[idx][:, 0], datagen.X[idx][:, 1], c=color[l], s=0.1)
        idx = np.where(datagen.y == l + datagen.offset)
        plt.scatter(datagen.X[idx][:, 0], datagen.X[idx][:, 1], c=color[l + datagen.offset], s=0.1)
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.savefig("plots/hmc_toy_gaussians_task{0}_{1}.png".format(task_idx, tag))
