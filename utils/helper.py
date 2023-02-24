import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from functools import reduce
from typing import Union
import torch
from torch import nn


def show_grad_cam(
    model,
    device,
    images,
    labels,
    predictions,
    target_layer,
    classes,
    use_cuda=True,
):
    """
    model = model,
    device = device,
    images = input images
    labels = correct classes for the images
    predictions = predictions for the images. If the desired gradcam is for the correct classes, pass labels here.
    target_layer = string representation of layer e.g. "layer3.1.conv2"
    classes = list of class labels
    """
    target_layers = [get_module_by_name(model, target_layer)]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    fig = plt.figure(figsize=(32, 32))

    plot_idx = 1
    for i in range(len(images)):
        input_tensor = images[i].unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(predictions[i])]
        rgb_img = denormalize(images[i].cpu().numpy().squeeze())
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Layout = 6 images per row - 2 * (original image, gradcam and visualization)
        ax = fig.add_subplot(len(images) / 2, 6, plot_idx, xticks=[], yticks=[])
        ax.imshow(rgb_img, cmap="gray")
        ax.set_title("True class: {}".format(classes[labels[i]]))
        plot_idx += 1

        ax = fig.add_subplot(len(images) / 2, 6, plot_idx, xticks=[], yticks=[])
        ax.imshow(grayscale_cam, cmap="gray")
        ax.set_title("GradCAM Output\nTarget class: {}".format(classes[predictions[i]]))
        plot_idx += 1

        ax = fig.add_subplot(len(images) / 2, 6, plot_idx, xticks=[], yticks=[])
        ax.imshow(visualization, cmap="gray")
        ax.set_title("Visualization\nTarget class: {}".format(classes[predictions[i]]))
        plot_idx += 1

    plt.tight_layout()
    plt.show()


def denormalize(img):
    channel_means = (0.4914, 0.4822, 0.4465)
    channel_stdevs = (0.2470, 0.2435, 0.2616)
    img = img.astype(dtype=np.float32)

    for i in range(img.shape[0]):
        img[i] = (img[i] * channel_stdevs[i]) + channel_means[i]

    return np.transpose(img, (1, 2, 0))


def show_training_images(train_loader, count, classes):
    images, labels = next(iter(train_loader))
    images = images[0:count]
    labels = labels[0:count]

    fig = plt.figure(figsize=(20, 10))
    for i in range(count):
        sub = fig.add_subplot(count / 5, 5, i + 1)
        npimg = denormalize(images[i].cpu().numpy().squeeze())
        plt.imshow(npimg, cmap="gray")
        sub.set_title("Correct class: {}".format(classes[labels[i]]))
    plt.tight_layout()
    plt.show()


def show_misclassified_images(images, predictions, labels, classes):
    fig = plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        sub = fig.add_subplot(len(images) / 5, 5, i + 1)
        image = images[i]
        npimg = denormalize(image.cpu().numpy().squeeze())
        plt.imshow(npimg, cmap="gray")
        predicted = classes[predictions[i]]
        correct = classes[labels[i]]
        sub.set_title(
            "Correct class: {}\nPredicted class: {}".format(correct, predicted)
        )
    plt.tight_layout()
    plt.show()


def show_losses_and_accuracies(trainer, tester, epochs):
    fig, ax = plt.subplots(2, 2)

    train_epoch_loss_linspace = np.linspace(0, epochs, len(trainer.train_losses))
    test_epoch_loss_linspace = np.linspace(0, epochs, len(tester.test_losses))
    train_epoch_acc_linspace = np.linspace(0, epochs, len(trainer.train_accuracies))
    test_epoch_acc_linspace = np.linspace(0, epochs, len(tester.test_accuracies))

    ax[0][0].set_xlabel("Epoch")
    ax[0][0].set_ylabel("Train Loss")
    ax[0][0].plot(train_epoch_loss_linspace, trainer.train_losses)
    ax[0][0].tick_params(axis="y", labelleft=True, labelright=True)

    ax[0][1].set_xlabel("Epoch")
    ax[0][1].set_ylabel("Test Loss")
    ax[0][1].plot(test_epoch_loss_linspace, tester.test_losses)
    ax[0][1].tick_params(axis="y", labelleft=True, labelright=True)

    ax[1][0].set_xlabel("Epoch")
    ax[1][0].set_ylabel("Train Accuracy")
    ax[1][0].plot(train_epoch_acc_linspace, trainer.train_accuracies)
    ax[1][0].tick_params(axis="y", labelleft=True, labelright=True)
    ax[1][0].yaxis.set_ticks(np.arange(0, 101, 5))

    ax[1][1].set_xlabel("Epoch")
    ax[1][1].set_ylabel("Test Accuracy")
    ax[1][1].plot(test_epoch_acc_linspace, tester.test_accuracies)
    ax[1][1].tick_params(axis="y", labelleft=True, labelright=True)
    ax[1][1].yaxis.set_ticks(np.arange(0, 101, 5))

    fig.set_size_inches(30, 10)
    plt.tight_layout()
    plt.show()


def get_module_by_name(module: Union[torch.Tensor, nn.Module], access_string: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def get_device():
    """
    Returns True, torch.device("cuda") if GPU is available
    else returns false, torch.device("cpu")
    """
    is_cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda_available else "cpu")
    return is_cuda_available, device


def show_lr_history(trainer, epochs):
    fig, ax = plt.subplots()

    linspace = np.linspace(0, epochs, len(trainer.lr_history))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.plot(linspace, trainer.lr_history)
    ax.tick_params(axis="y", labelleft=True, labelright=True)

    # fig.set_size_inches(30, 10)
    plt.tight_layout()
    plt.show()