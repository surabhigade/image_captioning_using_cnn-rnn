import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
from model import ImageCaptioningModel, get_transform
import json
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class FlickrDataset(Dataset):
    def __init__(self, image_dir, caption_file, vocab_file, transform=None):
        """Initialize the dataset."""
        self.root_dir = image_dir
        self.transform = transform or get_transform()
        
        # Load vocabulary
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
            self.word2idx = vocab_data['word2idx']
            self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        
        # Load captions
        with open(caption_file, 'r') as f:
            self.captions = json.load(f)
            
        # Create image-caption pairs
        self.image_caption_pairs = []
        for img_name, caps in self.captions.items():
            img_path = os.path.join(self.root_dir, img_name)
            for cap in caps:
                if os.path.exists(img_path):
                    self.image_caption_pairs.append((img_path, cap))

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        img_path, caption = self.image_caption_pairs[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        # Convert caption to tensor of indices
        tokens = caption.split()
        caption = []
        caption.append(self.word2idx['<start>'])
        caption.extend([self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens])
        caption.append(self.word2idx['<end>'])
        
        return image, torch.tensor(caption)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption)."""
    # Sort a data list by caption length (descending order)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Stack images
    images = torch.stack(images, 0)
    
    # Get caption lengths
    lengths = torch.tensor([len(cap) for cap in captions])
    
    # Pad captions with zeros
    targets = pad_sequence(captions, batch_first=True, padding_value=0)
    
    return images, targets, lengths

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    n_batches = len(train_loader)
    
    with tqdm(total=n_batches, desc='Training') as pbar:
        for i, (images, captions, lengths) in enumerate(train_loader):
            # Move to device
            images = images.to(device)
            captions = captions.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            caption_input = captions[:, :-1]
            caption_target = captions[:, 1:]
            caption_lengths = [l - 1 for l in lengths.tolist()]
            
            # Prepare lengths and forward pass
            
            outputs = model(images, caption_input, caption_lengths)
            packed_targets = pack_padded_sequence(caption_target, caption_lengths, batch_first=True, enforce_sorted=True)
            outputs_flat = outputs.view(-1, outputs.size(-1))
            
            
            # Calculate loss
            loss = criterion(outputs, packed_targets.data)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
    
    return total_loss / n_batches

def validate(model, val_loader, criterion, device):
    """Validates the model."""
    model.eval()
    total_loss = 0
    n_batches = len(val_loader)
    
    with torch.no_grad():
        with tqdm(total=n_batches, desc='Validating') as pbar:
            for i, (images, captions, lengths) in enumerate(val_loader):
                # Move to device
                images = images.to(device)
                captions = captions.to(device)
                
                caption_input = captions[:, :-1]
                caption_target = captions[:, 1:]
                caption_lengths = [l - 1 for l in lengths.tolist()]

                outputs = model(images, caption_input, caption_lengths)
                packed_targets = pack_padded_sequence(caption_target, caption_lengths, batch_first=True, enforce_sorted=True)

                loss = criterion(outputs, packed_targets.data)
                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
    
    return total_loss / n_batches

def plot_losses(train_losses, val_losses):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/loss_plot.png')
    plt.close()

def main():
    # Parameters
    embed_size = 256
    hidden_size = 512
    num_layers = 2
    num_epochs = 30
    batch_size = 32
    learning_rate = 3e-4
    
    # Paths
    data_dir = os.path.join('data', 'processed')
    image_dir = os.path.join(data_dir, 'images')
    vocab_file = os.path.join(data_dir, 'vocab.json')
    train_caption_file = os.path.join(data_dir, 'train_captions.json')
    val_caption_file = os.path.join(data_dir, 'val_captions.json')
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    with open(vocab_file, 'r') as f:
        vocab_data = json.load(f)
        vocab_size = len(vocab_data['word2idx'])
    print(f"Vocabulary size: {vocab_size}")
    
    # Create data loaders
    train_dataset = FlickrDataset(
        image_dir=image_dir,
        caption_file=train_caption_file,
        vocab_file=vocab_file
    )
    
    val_dataset = FlickrDataset(
        image_dir=image_dir,
        caption_file=val_caption_file,
        vocab_file=vocab_file
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    model = ImageCaptioningModel(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers
    ).to(device)
      # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding index
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(model_dir, 'best_model.pth'))
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Plot losses
    plot_losses(train_losses, val_losses)

if __name__ == '__main__':
    main()

