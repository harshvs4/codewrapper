from torch.utils.data import DataLoader
from torchvision import datasets


class Cifar10DataLoader:
    def __init__(
        self,
        transforms,
        batch_size=128,
        is_cuda_available=False,
        shuffle=True,
    ) -> None:
        self.batch_size = batch_size
        self.dataloader_args = {"shuffle": shuffle, "batch_size": self.batch_size}
        self.means = [0.4914, 0.4822, 0.4465]
        self.stds = [0.2470, 0.2435, 0.2616]

        if is_cuda_available:
            self.dataloader_args["num_workers"] = 2
            self.dataloader_args["pin_memory"] = True

        self.classes = (
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        self.transforms = transforms

    def get_dataset(self, train=True):
        return datasets.CIFAR10(
            "./data",
            train=train,
            transform=self.transforms(self.means, self.stds, train),
            download=True,
        )

    def get_loader(self, train=True):
        return DataLoader(self.get_dataset(train), **self.dataloader_args)

    def get_classes(self):
        return self.classes
