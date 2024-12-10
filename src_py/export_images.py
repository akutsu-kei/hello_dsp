import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from transform import transform_test

def export_image_to_header(image_path, header_file, label):
    image = Image.open(image_path)
    image = transform_test(image).unsqueeze(0) 

    assert torch.max(image) <= 1.0
    assert torch.min(image) >= -1.0

    scale_factor = 2**15
    
    with open(header_file, 'a') as f:
        f.write('const q15_t image_' + str(label) + '[] = {\n\t')
        image = image * scale_factor
        image = torch.clamp(image, -scale_factor, scale_factor - 1)
        data_list = [f"{round(x.item())}" for x in image.flatten()]
        f.write(", ".join(data_list))
        f.write("\n};\n\n")


if __name__ == '__main__':
    image_pathes = [
        r"mnist_images\test\0\3.png",
        r"mnist_images\test\1\2.png",
        r"mnist_images\test\2\1.png",
        r"mnist_images\test\3\18.png",
        r"mnist_images\test\4\4.png",
        r"mnist_images\test\5\15.png",
        r"mnist_images\test\6\11.png",
        r"mnist_images\test\7\0.png",
        r"mnist_images\test\8\61.png",
        r"mnist_images\test\9\7.png"
    ]

    labels = [
        0, 
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9
    ]

    path_to_header = "../image_data.h"
    with open(path_to_header, 'w') as f:
        f.write('#include <arm_math_types.h>\n\n')

    for image_path, label in zip(image_pathes, labels):
        export_image_to_header(image_path, path_to_header, label)