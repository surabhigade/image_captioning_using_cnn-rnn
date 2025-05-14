import torch
import torch.nn as nn
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import Counter
from model import ImageCaptioningModel, get_transform
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from tqdm import tqdm

# Create output directories
os.makedirs('outputs/evaluation_metrics', exist_ok=True)
os.makedirs('outputs/generated_captions', exist_ok=True)

class CaptionGenerator:
    def __init__(self, model_path, vocab_path, embed_size=256, hidden_size=512, num_layers=1):
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = ImageCaptioningModel(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=len(self.word2idx),
            num_layers=num_layers
        ).to(self.device)
          # Load model parameters from checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # If it's a training checkpoint
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            # If it's just the state dict
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Image transform
        self.transform = get_transform()
    
    def generate_caption(self, image_path):
        """Generate caption for an image."""
        # Prepare image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # Generate caption
        with torch.no_grad():
            sampled_ids = self.model.sample(image)
        
        # Convert indices to words
        sampled_ids = sampled_ids[0].cpu().numpy()
        
        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = self.idx2word[word_id]
            if word == '<end>':
                break
            if word not in ['<start>', '<pad>']:
                sampled_caption.append(word)
        
        return ' '.join(sampled_caption)

def evaluate_model(generator, test_image_file):
    """Evaluate the model on test images."""
    with open(test_image_file, 'r') as f:
        test_images = f.readlines()
    
    results = []
    for img_name in test_images:
        img_name = img_name.strip()
        img_path = os.path.join('data', 'images', img_name)
        
        if os.path.exists(img_path):
            caption = generator.generate_caption(img_path)
            results.append({
                'image': img_name,
                'generated_caption': caption
            })
    
    return results

def evaluate_metrics(generator, test_image_file, caption_file):
    """Evaluate the model using BLEU scores."""
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('punkt')
    
    # Load reference captions
    with open(caption_file, 'r') as f:
        captions_data = json.load(f)
    
    # Get test images
    with open(test_image_file, 'r') as f:
        test_images = [line.strip() for line in f.readlines()]
    
    references = []
    hypotheses = []
    results = []
    
    print("Generating captions for test images...")
    for img_name in tqdm(test_images):
        img_path = os.path.join('data', 'images', img_name)
        
        if os.path.exists(img_path) and img_name in captions_data:
            # Generate caption
            generated_caption = generator.generate_caption(img_path)
            hypothesis = word_tokenize(generated_caption.lower())
            hypotheses.append(hypothesis)
            
            # Get reference captions
            ref_captions = captions_data[img_name]
            ref_tokens = [word_tokenize(cap.lower()) for cap in ref_captions]
            references.append(ref_tokens)
            
            # Calculate individual BLEU scores
            individual_scores = {
                'bleu1': sentence_bleu(ref_tokens, hypothesis, weights=(1.0, 0, 0, 0)) * 100,
                'bleu2': sentence_bleu(ref_tokens, hypothesis, weights=(0.5, 0.5, 0, 0)) * 100,
                'bleu3': sentence_bleu(ref_tokens, hypothesis, weights=(0.33, 0.33, 0.33, 0)) * 100,
                'bleu4': sentence_bleu(ref_tokens, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)) * 100
            }
            
            results.append({
                'image': img_name,
                'generated_caption': generated_caption,
                'reference_captions': ref_captions,
                'scores': individual_scores
            })
    
    # Calculate corpus-level BLEU scores
    corpus_scores = {
        'bleu1': corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0)) * 100,
        'bleu2': corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0)) * 100,
        'bleu3': corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0)) * 100,
        'bleu4': corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)) * 100
    }
    
    return results, corpus_scores

def visualize_predictions(results, num_samples=5):
    """Visualize predictions with highest BLEU scores."""
    import numpy as np
    from PIL import Image
    
    # Sort results by average BLEU score
    def avg_bleu(sample):
        if 'scores' not in sample:
            return 0
        scores = sample['scores']
        return np.mean([scores['bleu1'], scores['bleu2'], scores['bleu3'], scores['bleu4']])
    
    sorted_results = sorted(results, key=avg_bleu, reverse=True)
    samples = sorted_results[:num_samples]  # Take top scoring samples
    
    fig = plt.figure(figsize=(15, 5*num_samples))
    for i, sample in enumerate(samples):
        # Load and display image
        img_path = os.path.join('data', 'images', sample['image'])
        image = Image.open(img_path).convert('RGB')
        
        ax = fig.add_subplot(num_samples, 1, i+1)
        ax.imshow(image)
        ax.axis('off')
        
        # Display captions and scores
        title = f"Generated: {sample['generated_caption']}\n"
        if 'scores' in sample:
            title += f"BLEU Scores: "
            title += f"B1={sample['scores']['bleu1']:.2f}, "
            title += f"B2={sample['scores']['bleu2']:.2f}, "
            title += f"B3={sample['scores']['bleu3']:.2f}, "
            title += f"B4={sample['scores']['bleu4']:.2f}\n"
            title += f"Average BLEU: {avg_bleu(sample):.2f}\n"
        title += "References:\n"
        for j, ref in enumerate(sample['reference_captions'], 1):
            title += f"{j}. {ref}\n"
        ax.set_title(title, fontsize=10, loc='left')
    
    plt.tight_layout()
    plt.savefig('outputs/evaluation_metrics/top_predictions.png')  # Changed filename
    plt.close()

def plot_bleu_scores_distribution(results):
    """Plot BLEU scores for each image in the test set."""
    # Extract image names and scores
    images = []
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    
    for result in results:
        if 'scores' in result:
            images.append(result['image'])
            bleu1_scores.append(result['scores']['bleu1'])
            bleu2_scores.append(result['scores']['bleu2'])
            bleu3_scores.append(result['scores']['bleu3'])
            bleu4_scores.append(result['scores']['bleu4'])
    
    # Create figure
    plt.figure(figsize=(20, 10))
    
    # Plot all BLEU scores
    plt.plot(range(len(images)), bleu1_scores, 'b.', label='BLEU-1', alpha=0.5)
    plt.plot(range(len(images)), bleu2_scores, 'r.', label='BLEU-2', alpha=0.5)
    plt.plot(range(len(images)), bleu3_scores, 'g.', label='BLEU-3', alpha=0.5)
    plt.plot(range(len(images)), bleu4_scores, 'y.', label='BLEU-4', alpha=0.5)
    
    # Customize the plot
    plt.title('BLEU Scores Distribution Across Test Images')
    plt.xlabel('Images')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add mean score lines
    mean_bleu1 = np.mean(bleu1_scores)
    mean_bleu2 = np.mean(bleu2_scores)
    mean_bleu3 = np.mean(bleu3_scores)
    mean_bleu4 = np.mean(bleu4_scores)
    
    plt.axhline(y=mean_bleu1, color='b', linestyle='--', alpha=0.5, label=f'Mean BLEU-1: {mean_bleu1:.2f}')
    plt.axhline(y=mean_bleu2, color='r', linestyle='--', alpha=0.5, label=f'Mean BLEU-2: {mean_bleu2:.2f}')
    plt.axhline(y=mean_bleu3, color='g', linestyle='--', alpha=0.5, label=f'Mean BLEU-3: {mean_bleu3:.2f}')
    plt.axhline(y=mean_bleu4, color='y', linestyle='--', alpha=0.5, label=f'Mean BLEU-4: {mean_bleu4:.2f}')
    
    # Update legend with mean scores
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('outputs/evaluation_metrics/bleu_scores_by_image.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_frequent_words(results, top_n=50):
    """Plot top N frequent words in predicted captions."""
    import nltk
    
    # Combine all predicted captions
    all_words = []
    for result in results:
        words = nltk.word_tokenize(result['generated_caption'].lower())
        all_words.extend(words)
    
    # Count word frequencies
    word_freq = Counter(all_words)
    
    # Get top N words
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    plt.figure(figsize=(15, 8))
    plt.bar(range(len(top_words)), list(top_words.values()))
    plt.xticks(range(len(top_words)), list(top_words.keys()), rotation=45, ha='right')
    plt.title(f'Top {top_n} Most Frequent Words in Generated Captions')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    plt.savefig('outputs/evaluation_metrics/frequent_words.png')
    plt.close()

def visualize_low_scoring_predictions(results, num_samples=5):
    """Visualize predictions with lowest BLEU scores."""
    import numpy as np
    from PIL import Image
    
    # Sort results by average BLEU score (ascending)
    def avg_bleu(sample):
        if 'scores' not in sample:
            return 0
        scores = sample['scores']
        return np.mean([scores['bleu1'], scores['bleu2'], scores['bleu3'], scores['bleu4']])
    
    sorted_results = sorted(results, key=avg_bleu)  # Sort in ascending order
    samples = sorted_results[:num_samples]  # Take bottom scoring samples
    
    fig = plt.figure(figsize=(15, 5*num_samples))
    for i, sample in enumerate(samples):
        # Load and display image
        img_path = os.path.join('data', 'images', sample['image'])
        image = Image.open(img_path).convert('RGB')
        
        ax = fig.add_subplot(num_samples, 1, i+1)
        ax.imshow(image)
        ax.axis('off')
        
        # Display captions and scores
        title = f"Generated: {sample['generated_caption']}\n"
        if 'scores' in sample:
            scores = sample['scores']
            title += f"BLEU Scores: B1={scores['bleu1']:.2f}, B2={scores['bleu2']:.2f}, "
            title += f"B3={scores['bleu3']:.2f}, B4={scores['bleu4']:.2f}\n"
            title += f"Average BLEU: {avg_bleu(sample):.2f}\n"
        
        title += "References:\n"
        for j, ref in enumerate(sample['reference_captions'], 1):
            title += f"{j}. {ref}\n"
        
        ax.set_title(title, fontsize=10, loc='left')
    
    plt.tight_layout()
    plt.savefig('outputs/evaluation_metrics/lowest_predictions.png')
    plt.close()

def main():
    # Model configuration
    model_path = os.path.join('models', 'best_model.pth')
    vocab_path = os.path.join('data', 'processed', 'vocab.json')
    test_image_file = os.path.join('data', 'Flickr_8k.testImages.txt')
    test_caption_file = os.path.join('data', 'processed', 'test_captions.json')
    
    print("Initializing caption generator...")
    generator = CaptionGenerator(
        model_path=model_path,
        vocab_path=vocab_path,
        embed_size=256,
        hidden_size=512,
        num_layers=2
    )
    
    print("Evaluating model...")
    results, metrics = evaluate_metrics(generator, test_image_file, test_caption_file)
    
    # Save results and metrics
    os.makedirs('outputs/evaluation_metrics', exist_ok=True)
    os.makedirs('outputs/generated_captions', exist_ok=True)
    
    with open('outputs/generated_captions/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open('outputs/evaluation_metrics/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.2f}")
    
    print("\nGenerating visualizations...")
    
    # Plot BLEU scores distribution
    plot_bleu_scores_distribution(results)
    print("- BLEU scores distribution plot saved")
    
    # Plot frequent words
    plot_frequent_words(results, top_n=50)
    print("- Frequent words plot saved")
    
    # Visualize best and worst predictions
    visualize_predictions(results, num_samples=5)
    print("- Top-scoring predictions visualization saved")
    
    visualize_low_scoring_predictions(results, num_samples=5)
    print("- Low-scoring predictions visualization saved")
    
    print("\nTesting completed!")
    print("Results saved in outputs/evaluation_metrics/ directory")

if __name__ == '__main__':
    main()

