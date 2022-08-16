import os, sys, json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class GarbageDataSet(Dataset):
    IMG_SIZE = 32 # For ResNet32
    C = 48
    CLASS_HIERARCHY = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=(IMG_SIZE,IMG_SIZE),scale=(0.8,1.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.05),
        transforms.RandomRotation(30),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
    ])
    
    def __init__(self, train=True):
        #定义好 image 的路径
        self.train=train
        self.images, self.targets = self.load_all_images()
        t_images = np.zeros((len(self.images), 32, 32, 3), dtype=np.uint8)
        for i, img in enumerate(self.images):
            if i % 100 == 0:
                print("\r Loading Images... {}".format(i), end="", flush=True)
            img = Image.open(img)
            img = img.convert("RGB")
            t_images[i] = img
        self.images = t_images
            
    def load_all_images(self):
        root_path = "F:\\Datasets\\垃圾目录\\class44_data"
        if GarbageDataSet.IMG_SIZE == 32:
            root_path = "F:\\Datasets\\垃圾目录\\class44_data_32x32"
        if self.train:
            root_path = os.path.join(root_path, "train")
        else:
            root_path = os.path.join(root_path, "val")
        imgs = []
        targets = []
        class_mapping = json.load(open("class_mapping_hm.json", "r", encoding="utf-8"))
        subclasses2main = json.load(open("subclass2main.json", "r", encoding="utf-8"))
        
        classes = os.listdir(root_path)
        for c in classes:
            c_imgs = os.listdir(os.path.join(root_path, c))
            c_imgs = [os.path.join(root_path, c, img) for img in c_imgs]
            c_labels = np.zeros(48)
            c_labels[class_mapping[c]] = 1 # current class
            c_labels[class_mapping[subclasses2main[c]]] = 1 # corresponding parent class
            c_labels = np.tile(c_labels, reps=(len(c_imgs), 1))
            imgs.extend(c_imgs)
            targets.extend(list(c_labels))
        return imgs, targets   
        
    def __getitem__(self, index):
        img = self.images[index]
        if self.train:
            img = GarbageDataSet.transform_train(img)
        else:
            img = GarbageDataSet.transform_test(img)
        target = self.targets[index]
        return img,target

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    d = GarbageDataSet(train=True)
    imgs = []
    for i in range(len(d)):
        if i % 100 == 0:
            print("\r {}".format(i), end="", flush=True)
        imgs.append(d[i])
    
    print(len(imgs))