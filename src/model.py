import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.transforms as transforms

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained VGG16 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # Freeze all parameters
        for param in vgg16.parameters():
            param.requires_grad = False
            
        # Remove the last fc layer and add our own
        modules = list(vgg16.children())[:-1]
        self.vgg16 = nn.Sequential(*modules)
        self.linear = nn.Linear(512 * 7 * 7, embed_size)  # 512 is the number of channels in last conv layer
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.vgg16(images)
        features = features.reshape(features.size(0), -1)
        features = self.dropout(features)
        features = self.linear(features)
        features = self.bn(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.5 if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.max_seg_length = 20
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = self.dropout(embeddings)
        features = features.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        embeddings = embeddings + features  # Using additive features instead of concatenation
          # Pack the sequence
        lengths = torch.tensor(lengths).to('cpu')  # lengths must be on CPU
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=True)
        
        # LSTM forward pass
        hiddens, _ = self.lstm(packed)
        
        # Get the output
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None, max_length=20):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        
        for i in range(max_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted)
            
            if predicted.item() == 2:  # <end> token
                break
                
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """Initialize the model."""
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, images, captions, lengths):
        """Forward pass through the model."""
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs
    
    def sample(self, image, states=None):
        """Generate caption for an image."""
        features = self.encoder(image)
        return self.decoder.sample(features, states)

def get_transform():
    """Returns the image preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

