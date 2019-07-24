import torch
import torchvision
from torchvision import transforms, datasets
import os
from PIL import Image
import numpy as np
import sys
import argparse
import resnet
import io
import cv2

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = resnet.ResNet(resnet.Bottleneck, [3,8,36,3])
    model.load_state_dict(torch.load('resnet152_best-148.ckpt', map_location=device))

    loader = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    model.cuda()
    model.eval()

    with torch.no_grad():
        # ResNet152 Python Pytorch Test...
        '''
        img_path = "9/" + i

        image = Image.open(img_path)
        image = image.convert('RGB')

        image_tensor = loader(image).unsqueeze(0)
        image_numpy = image_tensor.numpy()
        
        image_tensor = image_tensor.cuda()

        output = model(image_tensor)
        _, predict = torch.max(output.data, 1)
        print("predicted : ", predict, i)
        '''
        
        # Save Script Module
        example = torch.rand(1,3,32,32).cuda()
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save('script_module.pt')


if __name__ == '__main__':
    main()
