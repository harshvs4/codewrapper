import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary


# Global reference
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def idx_to_class(idx):
    return classes[idx]

def model_summary(model, input_size=(3,32,32)):
    return summary(model, input_size=input_size)


def plot_stats(train_stats, test_stats, labels,xlabel=None, ylabel=None, title=None):
    plt.plot(train_stats, label=labels[0])
    plt.plot(test_stats, label=labels[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()



def plot_imgs(misclassified_imgs_dict, imgs=20):
    fig, axs = plt.subplots(5, 4, figsize=(10, 10), squeeze=False)
    fig.tight_layout(h_pad=2)
    idx = 0
    key_list = iter(list(misclassified_imgs_dict.keys()))

    for i in range(5):
        for j in range(4):
            idx = next(key_list)
            img = misclassified_imgs_dict[idx][0]
            true_label = idx_to_class(misclassified_imgs_dict[idx][1])
            pred_label = idx_to_class(misclassified_imgs_dict[idx][2].item())

            axs[i, j].imshow(np.transpose(img, (1,2,0)), cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title(f'True Label: {true_label}\n Pred Label:{pred_label}')