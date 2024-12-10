import os
from torchvision import datasets, transforms
from PIL import Image

def save_mnist_images(dataset, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for index, (image, label) in enumerate(dataset):
        label_dir = os.path.join(output_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        image = transforms.ToPILImage()(image)
        image_path = os.path.join(label_dir, f"{index}.png")
        image.save(image_path)

def download_mnist_images():
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)

    save_mnist_images(train_dataset, './mnist_images/train')
    save_mnist_images(test_dataset, './mnist_images/test')

if __name__ == '__main__':
    download_mnist_images()