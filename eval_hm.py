import os,json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from effnet import efficientnetv2_s
from resnet import ResNet, BasicBlock
import PIL
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
np.set_printoptions(suppress=True)

# Global parameters
n_classes = 48
image_path = "test_images/pillow_thumbnails.jpg"
class_mapping = json.load(open("class_mapping_hm.json", "r", encoding="utf-8"))
H = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
HH = [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]
image_class = np.zeros(48)
image_class[H[class_mapping["枕头"]]] = 1
image_class[class_mapping["枕头"]] = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取数据
# 对应文件夹的label
model = ResNet(BasicBlock, [5, 5, 5], num_classes=n_classes)
# 初始化 https://androidkt.com/initialize-weight-bias-pytorch/
model.linear.bias.data.fill_(-np.log(n_classes-1))
ckpt_file_path = os.path.join("models/best_model_lps-mgl.pth")
model.load_state_dict(torch.load(ckpt_file_path))
model.to(DEVICE)

#验证过程
model.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
])
with torch.no_grad():
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = transform(image)[None, ...].to(DEVICE)
    output = model(image)
    for i, hh in enumerate(HH):
        probs = F.softmax(output.data[..., hh], dim=1)
        print(probs.cpu().numpy()[0])
        _, pred = torch.max(probs, dim=1)
        print(pred.cpu().numpy()[0])
        label = np.argmax(image_class[hh])
        print(label)
    