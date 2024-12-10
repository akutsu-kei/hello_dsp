import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from PIL import Image

from model import MLP
from transform import transform_test

def test_by_file(image_path):
    model = MLP()
    model.load_state_dict(torch.load('mlp_mnist.pth'))
    model.eval()
    
    image = Image.open(image_path)
    image = transform_test(image).unsqueeze(0)  

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        print(f'Judgement: {predicted.item()}')

    return predicted.item()

def test():
    assert test_by_file(r"mnist_images\test\0\3.png") == 0
    assert test_by_file(r"mnist_images\test\1\2.png") == 1
    assert test_by_file(r"mnist_images\test\2\1.png") == 2
    assert test_by_file(r"mnist_images\test\3\18.png") == 3
    assert test_by_file(r"mnist_images\test\4\4.png") == 4
    assert test_by_file(r"mnist_images\test\5\15.png") == 5
    assert test_by_file(r"mnist_images\test\6\11.png") == 6
    assert test_by_file(r"mnist_images\test\7\0.png") == 7
    assert test_by_file(r"mnist_images\test\8\61.png") == 8
    assert test_by_file(r"mnist_images\test\9\7.png") == 9

def export_test_images(output_path):
    model = MLP()
    model.load_state_dict(torch.load('mlp_mnist.pth'))
    model.eval()

    scale_factor = 2**15

    with open(output_path, 'w') as f:
        f.write("#include <arm_math.h>\n\n")
        f.write(f"#define INPUT_SIZE {model.fc1.in_features}\n")
        f.write(f"#define FC1_SIZE {model.fc1.out_features}\n\n")
        f.write(f"#define FC2_SIZE {model.fc2.out_features}\n\n")

        for name, param in model.named_parameters():
            data = param.data.transpose(0, 1)
            assert torch.min(data) > -1.0
            assert torch.max(data) < 1.0
            data = data * scale_factor
            data = torch.clamp(data, -scale_factor, scale_factor - 1)
            name_fixed = name.replace('.', '_')
            f.write(f"const q15_t {name_fixed}[] = {{\n\t")
            data_list = [f"{round(x.item())}" for x in data.flatten()]
            f.write(", ".join(data_list))
            f.write("\n};\n\n")


if __name__ == '__main__':
    test()
    print("All tests passed!")
    export_test_images('../model_parameters.h')
    print("Model parameters exported to ../model_parameters.h")
