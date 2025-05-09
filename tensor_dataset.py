import os
import numpy as np
import torch
from glob import glob
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from torch.utils.data import Dataset, DataLoader

def load_data_and_compute_features(dataset_root): 
    # 1) Gather all file-paths and labels
    all_paths, all_labels = [], []
    label_map = {}
    for npy_file in glob(os.path.join(dataset_root, '**/*.npy'), recursive=True):
        cls = os.path.basename(os.path.dirname(npy_file))
        label = label_map.setdefault(cls, len(label_map))
        all_paths.append(npy_file)
        all_labels.append(label)

    # 2) Stratified split into train/dev/test (20% test, then 25% of remaining -> dev)
    print("\nSplitting into train/dev/test sets...")
    paths_temp, paths_test, labels_temp, labels_test = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    paths_train, paths_dev, labels_train, labels_dev = train_test_split(
        paths_temp, labels_temp, test_size=0.25, random_state=42, stratify=labels_temp
    )

    # 3) Define a Dataset that loads + normalizes on the fly
    class NpyFeatureDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = paths
            self.labels = labels
            self.transform = transform
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, i):
            img = np.load(self.paths[i])
            # if shape is H×W×3, convert to C×H×W
            if img.ndim == 3 and img.shape[-1] == 3 and img.shape[0] != 3:
                img = img.transpose(2,0,1)
            img = img.astype(np.uint8)                   # PIL wants uint8
            pil = Image.fromarray(img.transpose(1,2,0))  # C×H×W -> H×W×C
            return self.transform(pil), self.labels[i]

    # 4) Build your torchvision-style preprocess pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),           # shorter side -> 256, keep aspect
        transforms.CenterCrop(224),       # center 224×224
        transforms.ToTensor(),            # -> [0,1], C×H×W
        transforms.Normalize(             # ImageNet mean/std
        mean=[0.485,0.456,0.406],
        std =[0.229,0.224,0.225],
        ),
    ])

    # 5) Datasets & DataLoaders
    train_ds = NpyFeatureDataset(paths_train, labels_train, preprocess)
    dev_ds   = NpyFeatureDataset(paths_dev,   labels_dev,   preprocess)
    test_ds  = NpyFeatureDataset(paths_test,  labels_test,  preprocess)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0, pin_memory=True)


    # # 1) Inspect the raw .npy on disk
    # path0 = paths_train[0]
    # arr0  = np.load(path0)
    # print("On-disk array shape:", arr0.shape)   # e.g. (H, W, 3)

    # # 2) Inspect what your Dataset returns  
    # img_tensor0, label0 = train_ds[0]
    # print("Tensor from Dataset:", img_tensor0.shape, "label:", label0)
    # # should be torch.Size([3,224,224]) and an int

    # # 3) Inspect one batch from the DataLoader  
    # batch_imgs, batch_lbls = next(iter(train_loader))
    # print("Batch images:", batch_imgs.shape)   # e.g. torch.Size([64,3,224,224])
    # print("Batch labels:", batch_lbls.shape)   # e.g. torch.Size([64])
    # print("Min / Max (batch):", batch_imgs.min().item(), batch_imgs.max().item())

    print("\nExtracting features using ResNet50...")
    # 6) Load ResNet50, strip its head, run inference, collect features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.to(device).eval()

    def extract_feats(loader):
        all_feats, all_lbls = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                feats = model(imgs)
                all_feats.append(feats.cpu())
                all_lbls.append(lbls)
        return torch.cat(all_feats, dim=0), torch.cat(all_lbls, dim=0)

    X_train, y_train = extract_feats(train_loader)
    X_dev,   y_dev   = extract_feats(dev_loader)
    X_test,  y_test  = extract_feats(test_loader)

    # 7) Save everything to single .pt for later
    torch.save({
        'X_train': X_train, 'y_train': y_train,
        'X_dev':   X_dev,   'y_dev':   y_dev,
        'X_test':  X_test,  'y_test':  y_test
    }, '../normalized_data.pt')

    print("Feature extraction complete!")


dataset_root = '../dataset_images/dataset'
load_data_and_compute_features(dataset_root)