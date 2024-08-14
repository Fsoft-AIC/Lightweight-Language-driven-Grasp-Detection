# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: unsloth_env
#     language: python
#     name: python3
# ---

import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Dataset
import alpha_clip
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import argparse

# Create arguments parser
parser = argparse.ArgumentParser(description='Train robotic clip model')
parser.add_argument('--dataset_path', type=str, default='data/something/training', help='Path to the dataset')
parser.add_argument('--checkpoint', type=str, default='checkpoints/clip_l14_336_grit_20m_4xe.pth', help='Path to the checkpoint')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--clip_model', type=str, default='ViT-L/14@336px', help='CLIP model')
parser.add_argument('--frac', type=float, default=0.9, help='Fraction of the dataset to use for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--robotic_checkpoint', type=str, default='checkpoints/robotic_clip/', help='Path to the robotic clip checkpoint')
parser.add_argument('--limit', type=int, default=1000, help='Limit the number of samples in the dataset')
parser.add_argument('--train_batch_per_epoch', type=int, default=10000, help='Random seed')
parser.add_argument('--val_batch_per_epoch', type=int, default=1000, help='Random seed')
parser.add_argument('--gamma', type=float, default=0.1, help='Triplet loss weight')

args = parser.parse_args()

dataset_path = args.dataset_path
checkpoint = args.checkpoint
batch_size = args.batch_size
clip_model = args.clip_model
frac = args.frac
epochs = args.epochs
robotic_checkpoint = args.robotic_checkpoint
limit = args.limit
train_batch_per_epoch = args.train_batch_per_epoch
val_batch_per_epoch = args.val_batch_per_epoch
gamma = args.gamma

class Robotic_Dataset(Dataset):
    def __init__(self, dataset_path, preprocess, mask_transform, limit=None):
        self.dataset_path = dataset_path
        self.dataset = os.listdir(os.path.join(dataset_path, 'prompt'))
        if limit:
            if limit < len(self.dataset):
                self.dataset = np.random.choice(self.dataset, limit)
        self.preprocess = preprocess
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        id = self.dataset[idx].split('.')[0]
        img_start_pth = os.path.join(self.dataset_path, 'image', id + '_0.jpg')
        img_end_pth = os.path.join(self.dataset_path, 'image', id + '_1.jpg')
        mask_start_pth = os.path.join(self.dataset_path, 'mask', id + '_0.npy')
        mask_end_pth = os.path.join(self.dataset_path, 'mask', id + '_1.npy')
        image_start = Image.open(img_start_pth).convert('RGB')
        image_end = Image.open(img_end_pth).convert('RGB')
        mask_start = np.load(mask_start_pth) 
        mask_end = np.load(mask_end_pth)
        
        alpha_start = self.mask_transform((mask_start * 255).astype(np.uint8))
        alpha_end = self.mask_transform((mask_end * 255).astype(np.uint8))
        image_start = self.preprocess(image_start)
        image_end = self.preprocess(image_end)
        
        with open(os.path.join(self.dataset_path, 'prompt', id + '.txt')) as f:
            prompt = f.read()
            
        return image_start, image_end, alpha_start, alpha_end, prompt


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cud"
model, preprocess = alpha_clip.load(clip_model, alpha_vision_ckpt_pth=checkpoint, device=device)

if clip_model == 'ViT-L/14@336px':
    size = 336
else:
    size = 224

mask_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((size, size)), # change to (336,336) when using ViT-L/14@336px
    transforms.Normalize(0.5, 0.26)
])

dataset = Robotic_Dataset(dataset_path, preprocess, mask_transform, limit = limit)

# Split the dataset into training and validation
train_size = int(frac * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()
    
torch.autograd.set_detect_anomaly(True)
    
criterion_image = nn.CrossEntropyLoss()
criterion_text = nn.CrossEntropyLoss()
criterion_triplet = TripletLoss(margin=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

os.makedirs(robotic_checkpoint, exist_ok=True)
best_val_loss = 100000

for epoch in range(epochs):
    model.train()
    for i in tqdm(range(train_batch_per_epoch)):
        # Randomly sample a batch from DataLoader
        optimizer.zero_grad()
        image_start, image_end, alpha_start, alpha_end, prompt = next(iter(train_loader))
        text = alpha_clip.tokenize(prompt).to(device)
        image_start = image_start.half().to(device)
        image_end = image_end.half().to(device)
        alpha_start = alpha_start.half().to(device)
        alpha_end = alpha_end.half().to(device)
        image_features_start = model.visual(image_start, alpha_start)
        image_features_end = model.visual(image_end, alpha_end)
        text_features = model.encode_text(text)
        # Normalize the features
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # image_features_start = image_features_start / image_features_start.norm(dim=-1, keepdim=True)
        # image_features_end = image_features_end / image_features_end.norm(dim=-1, keepdim=True)
        logits_per_image1, logits_per_text1 = model(image_start, text, alpha_start)
        logits_per_image2, logits_per_text2 = model(image_end, text, alpha_end)
        # similarity1 = (100.0 * image_features_start @ text_features.T)
        # similarity2 = (100.0 * image_features_end @ text_features.T)
        ground_truth = torch.arange(batch_size).to(device)
        # loss_value = (criterion(similarity1, ground_truth) + criterion(similarity2, ground_truth))/2
        constrastive_loss = (criterion_image(logits_per_image1,ground_truth) + criterion_text(logits_per_text1,ground_truth) 
                      + criterion_image(logits_per_image2,ground_truth) + criterion_text(logits_per_text2,ground_truth))/4
        triplet_loss = criterion_triplet(text_features, image_features_end, image_features_start)
        loss_value = constrastive_loss + gamma*triplet_loss
        # loss_value = criterion(similarity2, ground_truth)
        loss_value.backward()

        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            alpha_clip.model.convert_weights(model)
        if i % 10 == 0:
            print(f'Epoch {epoch}, iter {i}, loss {loss_value.item()}')
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for i in tqdm(range(val_batch_per_epoch)):
            image_start, image_end, alpha_start, alpha_end, prompt = next(iter(val_loader))
            text = alpha_clip.tokenize(prompt).to(device)
            image_start = image_start.half().to(device)
            image_end = image_end.half().to(device)
            alpha_start = alpha_start.half().to(device)
            alpha_end = alpha_end.half().to(device)
            logits_per_image, logits_per_text = model(image_start, text, alpha_start)
            ground_truth = torch.arange(batch_size).to(device)
            # loss_value = (criterion(similarity1, ground_truth) + criterion(similarity2, ground_truth))/2
            loss_value = (criterion_image(logits_per_image1,ground_truth) + criterion_text(logits_per_text1,ground_truth) 
                      + criterion_image(logits_per_image2,ground_truth) + criterion_text(logits_per_text2,ground_truth))/4
            val_loss += loss_value.item()
        avg_val_loss = val_loss / val_batch_per_epoch
        print(f'RESULT: Epoch {epoch}, val_loss {avg_val_loss}')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(robotic_checkpoint, f'robotic_clip_{epoch}.pth'))
    lr_scheduler.step()