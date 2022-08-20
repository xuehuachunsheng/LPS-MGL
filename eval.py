# Train EfficientNetV2 from scratch
#from ossaudiodev import openmixer
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
from resnet import ResNet, BasicBlock
import PIL
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
np.set_printoptions(suppress=True)

# 读取数据
# 对应文件夹的label
n_classes = 44
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet(BasicBlock, [5, 5, 5], num_classes=n_classes)
# 初始化 https://androidkt.com/initialize-weight-bias-pytorch/
model.linear.bias.data.fill_(-np.log(n_classes-1))
ckpt_file_path = os.path.join("models/Garbage/best_model_cel.pth")
model.load_state_dict(torch.load(ckpt_file_path))
model.to(DEVICE)

# 设置全局参数
image_path = "test_images/pillow_thumbnails.jpg"
class_mapping = json.load(open("class_mapping_single.json", "r", encoding="utf-8"))
image_class = class_mapping["枕头"]

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
    image = transform(image)[None,...].to(DEVICE)
    output = model(image)
    probs = F.softmax(output.data, dim=1)        
    if np.max(probs[0].cpu().numpy()) < 0.5:
        print(image_path)    
        print(probs[0].cpu().numpy())
        _, pred = torch.max(probs, dim=1)
        print(pred.cpu().numpy()[0])
        print(image_class)
    
