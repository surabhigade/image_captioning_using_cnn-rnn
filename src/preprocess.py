import os
import pandas as pd
import nltk
from collections import Counter
import json
import shutil
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

def load_captions(caption_file):
    """Load and preprocess captions from the Flickr8k dataset."""
    captions = {}
    with open(caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Split the line by '#' to separate image name and caption
            parts = line.split('#')
            if len(parts) >= 2:
                # The image name is in the first part
                image_name = parts[0].strip()
                # The caption follows the first number after '#'
                caption = parts[1].split('\t', 1)
                if len(caption) == 2:
                    caption_text = caption[1].strip().lower()
                    
                    if image_name not in captions:
                        captions[image_name] = []
                    captions[image_name].append(caption_text)
    
    print(f"Found {len(captions)} images with captions")
    if len(captions) == 0:
        print("WARNING: No captions were loaded! First few lines of the file:")
        with open(caption_file, 'r', encoding='utf-8') as f:
            print(f.read(500))
    
    return captions

def build_vocabulary(captions, min_word_freq=4):
    """Build vocabulary from captions with word frequency threshold."""
    word_freq = Counter()
    
    for img_captions in captions.values():
        for caption in img_captions:
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            word_freq.update(tokens)
    
    # Create vocabulary with special tokens
    vocab = {
        'word2idx': {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3},
        'idx2word': {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
    }
    
    # Add words that meet the frequency threshold
    idx = 4
    for word, freq in word_freq.items():
        if freq >= min_word_freq:
            vocab['word2idx'][word] = idx
            vocab['idx2word'][idx] = word
            idx += 1
    
    return vocab

def process_images(image_dir, output_dir, size=(256, 256)):
    """Process and resize images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size)
    ])
    
    # Process each image
    for img_name in tqdm(os.listdir(image_dir)):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            try:
                img_path = os.path.join(image_dir, img_name)
                img = Image.open(img_path).convert('RGB')
                
                # Apply transformations
                img = transform(img)
                
                # Save processed image
                output_path = os.path.join(output_dir, img_name)
                img.save(output_path, 'JPEG')
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")

def create_data_splits(caption_file, split_files):
    """Create train/val/test splits based on Flickr8k split files."""
    splits = {}
    
    # Read split files
    for split_name, split_file in split_files.items():
        print(f"Reading {split_name} split file: {split_file}")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        with open(split_file, 'r') as f:
            images = [line.strip() for line in f.readlines()]
            splits[split_name] = set(images)
        print(f"Found {len(splits[split_name])} images in {split_name} split")
    
    print(f"Reading caption file: {caption_file}")
    if not os.path.exists(caption_file):
        raise FileNotFoundError(f"Caption file not found: {caption_file}")
        
    # Load all captions
    all_captions = load_captions(caption_file)
    print(f"Loaded {len(all_captions)} images with captions")
    
    # Create split dictionaries
    split_captions = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    # Assign captions to splits
    for img_name, captions in all_captions.items():
        if img_name in splits['train']:
            split_captions['train'][img_name] = captions
        elif img_name in splits['val']:
            split_captions['val'][img_name] = captions
        elif img_name in splits['test']:
            split_captions['test'][img_name] = captions
    
    return split_captions

def main():
    # Create processed data directory
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Download NLTK data if not already downloaded
    nltk.download('punkt')
    
    # Define paths
    caption_file = os.path.join('data', 'Flickr8k.token.txt')
    image_dir = os.path.join('data', 'images')
    processed_image_dir = os.path.join(processed_dir, 'images')
    
    split_files = {
        'train': os.path.join('data', 'Flickr_8k.trainImages.txt'),
        'val': os.path.join('data', 'Flickr_8k.devImages.txt'),
        'test': os.path.join('data', 'Flickr_8k.testImages.txt')
    }
    
    # Process images
    print("Processing images...")
    process_images(image_dir, processed_image_dir)
    
    # Load and preprocess captions
    print("Loading captions...")
    split_captions = create_data_splits(caption_file, split_files)
    
    # Build vocabulary from training captions
    print("Building vocabulary...")
    vocab = build_vocabulary(split_captions['train'])
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    
    # Save vocabulary
    vocab_file = os.path.join(processed_dir, 'vocab.json')
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    # Save split captions
    for split_name, captions in split_captions.items():
        output_file = os.path.join(processed_dir, f'{split_name}_captions.json')
        with open(output_file, 'w') as f:
            json.dump(captions, f, indent=2)
    
    print("Preprocessing completed!")
    print(f"Vocabulary size: {len(vocab['word2idx'])}")
    print(f"Number of training images: {len(split_captions['train'])}")
    print(f"Number of validation images: {len(split_captions['val'])}")
    print(f"Number of test images: {len(split_captions['test'])}")

if __name__ == '__main__':
    main()

