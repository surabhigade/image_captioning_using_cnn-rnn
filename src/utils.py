import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import os
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

def load_vocabulary(vocab_path):
    """Load vocabulary from json file."""
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    return vocab_data

def plot_training_progress(losses, save_path=None):
    """Plot training loss progress."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def calculate_metrics(references, hypotheses):
    """Calculate BLEU-1,2,3,4, METEOR, and ROUGE scores."""
    # Prepare references for BLEU score calculation
    ref_list = [[ref.split()] for ref in references]
    hyp_list = [hyp.split() for hyp in hypotheses]
    
    # Calculate BLEU scores
    bleu1 = corpus_bleu(ref_list, hyp_list, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(ref_list, hyp_list, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(ref_list, hyp_list, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(ref_list, hyp_list, weights=(0.25, 0.25, 0.25, 0.25))
    
    # Calculate METEOR score
    meteor = np.mean([meteor_score([ref], hyp) for ref, hyp in zip(references, hypotheses)])
    
    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    
    rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
    rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])
    
    return {
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4,
        'meteor': meteor,
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougeL': rougeL
    }

def save_metrics(metrics, output_file):
    """Save evaluation metrics to a json file."""
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

def visualize_attention(image_path, caption, attention_weights, vocab, save_path=None):
    """Visualize attention weights for image captioning."""
    # Load and resize image
    img = Image.open(image_path)
    plt.figure(figsize=(14, 7))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original Image')
    
    # Plot attention heatmap
    plt.subplot(1, 2, 2)
    att_map = attention_weights.reshape(attention_weights.shape[0], int(np.sqrt(attention_weights.shape[1])), -1)
    plt.imshow(att_map, cmap='hot')
    plt.colorbar()
    plt.title('Attention Weights')
    
    # Add word labels
    words = caption.split()
    plt.yticks(range(len(words)), words)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

class AverageMeter:
    """Keep track of average values."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

