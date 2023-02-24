#Importing Libraries
from codewrapper.utils.transforms import CustomResnetTransforms
from codewrapper.utils.load_data import Cifar10DataLoader
from codewrapper.utils.helper import *
from codewrapper.model import *
from codewrapper.utils.train import Train
from codewrapper.utils.test import Test
from torch_lr_finder import LRFinder
import numpy as np
import copy

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR


def save_model(model, epoch, optimizer, path):
    """Save torch model in .pt format

    Args:
        model (instace): torch instance of model to be saved
        epoch (int): epoch num
        optimizer (instance): torch optimizer
        path (str): model saving path
    """
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)


def train_model(train, test, NUM_EPOCHS, use_l1=False, scheduler=None, save_best=False):
    for epoch in range(1, NUM_EPOCHS + 1):
        train.train(epoch, scheduler)
        _, test_loss = test.test()

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)

        if save_best:
            min_val_loss = np.inf
            save_path = "model.pt"
            if test_loss < min_val_loss:
                print(
                    f"Valid loss reduced from {min_val_loss:.5f} to {test_loss:.6f}. checkpoint created at...{save_path}\n"
                )
                save_model(train.model, epoch, train.optimizer, save_path)
                min_val_loss = test_loss
            else:
                print(f"Valid loss did not inprove from {min_val_loss:.5f}")

        print()

    if scheduler:
        return train.model, (
            train.train_accuracies,
            train.train_losses,
            test.test_accuracies,
            test.test_losses,
            train.lr_history,
        )
    else:
        return train.model, (
            train.train_accuracies,
            train.train_losses,
            test.test_accuracies,
            test.test_losses,
        )


def get_lr(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    end_lr=10,
    num_iter=200,
    step_mode="exp",
    start_lr=None,
    diverge_th=5,
):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(
        train_loader,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode=step_mode,
        start_lr=start_lr,
        diverge_th=diverge_th,
    )
    lr_finder.plot()
    min_loss = min(lr_finder.history["loss"])
    max_lr = lr_finder.history["lr"][np.argmin(lr_finder.history["loss"], axis=0)]

    print("Min Loss = {}, Max LR = {}".format(min_loss, max_lr))

    # Reset the model and optimizer to initial state
    lr_finder.reset()

    return min_loss, max_lr


def run():
    is_cuda_available, device = get_device()
    cifar10 = Cifar10DataLoader(CustomResnetTransforms, 512, is_cuda_available)

    #print_summary(CustomResNet(), device, input_size=(3, 32, 32))
    result = summary(CustomResNet(), input_size=input_size)
    print(result)
    model = CustomResNet()

    train_loader = cifar10.get_loader(True)
    test_loader = cifar10.get_loader(False)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    min_loss, max_lr = get_lr(model, train_loader, optimizer, criterion, device)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=24,
        pct_start=5 / 24,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy="linear",
    )

    train = Train(model, train_loader, optimizer, criterion, device)
    test = Test(model, test_loader, criterion, device)

    train_model(train, test, NUM_EPOCHS=24, scheduler=scheduler)


if __name__ == "__main__":
    run()
