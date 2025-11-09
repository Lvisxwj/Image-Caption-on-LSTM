# Experiment 2: Image Captioning with LSTM

## Overview
This experiment implements an **Image Captioning** system using LSTM (Long Short-Term Memory) networks. The task is to generate natural language descriptions for images by combining computer vision (CNN feature extraction) and natural language processing (LSTM text generation).

## Project Structure

### Dataset
- **Location**: `Exp2/dataset/`
- **Images**: `Exp2/dataset/Images/` - Contains JPG images (e.g., `1000268201_693b08cb0e.jpg`)
- **Captions**: `Exp2/dataset/captions.txt` - CSV format with columns `image,caption`
  - Each image has **5 different captions** (diverse descriptions of the same image)
  - Example: "A child in a pink dress is climbing up a set of stairs in an entry way."

### Comparison with Exp1

| Aspect | Exp1 (MNIST) | Exp2 (Image Captioning) |
|--------|--------------|-------------------------|
| Task | Image Classification | Image-to-Text Generation |
| Input | 28x28 grayscale images | RGB images (variable size) |
| Output | Single label (0-9) | Text sequence (caption) |
| Model | CNN only | CNN (encoder) + LSTM (decoder) |
| Loss | CrossEntropyLoss | CrossEntropyLoss (per word) |
| Evaluation | Accuracy | BLEU score, Perplexity |

## Architecture Design

### 1. Encoder (CNN)
- **Purpose**: Extract visual features from images
- **Options**:
  - ResNet50/VGG16 (pre-trained on ImageNet)
  - Custom CNN architecture
- **Output**: Fixed-length feature vector (e.g., 2048-dim)

### 2. Decoder (LSTM)
- **Purpose**: Generate word sequences from visual features
- **Components**:
  - Embedding layer: Convert word indices to dense vectors
  - LSTM layers: Generate sequential predictions
  - Linear layer: Project to vocabulary size
- **Input**: Image features + previous words
- **Output**: Next word probabilities

### 3. Attention Mechanism (Optional Enhancement)
- Allow decoder to focus on different image regions when generating each word

## Implementation Pipeline (Following Exp1 Pattern)

### Cell 1: Data Loading & Preprocessing
```python
- Load captions.txt and images from dataset/
- Build vocabulary from captions (word tokenization)
- Create image transforms (resize, normalize)
- Split into train/validation sets (80/20)
- Create custom Dataset class:
  * __getitem__: return (image_tensor, caption_indices, caption_length)
- Create DataLoaders with appropriate batch_size
- Visualize sample images with captions
```

### Cell 2: Model Definition & Training
```python
- Define EncoderCNN class (feature extraction)
- Define DecoderLSTM class (caption generation)
- Set hyperparameters: EPOCHS, LR, EMBED_SIZE, HIDDEN_SIZE
- Handle device configuration (CUDA/CPU)
- Implement train_model function:
  * Teacher forcing during training
  * Track loss and perplexity
  * Save model checkpoints
- Train the model with progress monitoring
```

### Cell 3: Model Evaluation & Prediction
```python
- Load trained model
- Implement caption generation (greedy/beam search)
- Evaluate on validation set:
  * Calculate BLEU scores
  * Compute perplexity
- Generate predictions for sample images
- Export results to file
```

### Cell 4: Visualization
```python
- Plot training curves (loss, perplexity)
- Display sample predictions:
  * Show image
  * Display generated caption
  * Show ground truth captions (5 references)
- Create qualitative analysis grid
```

## Key Technical Considerations

### Vocabulary Building
- Add special tokens: `<PAD>`, `<START>`, `<END>`, `<UNK>`
- Filter low-frequency words (threshold: min_freq=5)
- Map words to indices for model input

### Caption Processing
- Tokenization: Split captions into words
- Add START/END tokens to each caption
- Pad sequences to same length within batch
- Handle variable-length sequences with pack_padded_sequence

### Training Strategy
- **Teacher Forcing**: Feed ground truth words during training (not predictions)
- **Scheduled Sampling**: Gradually transition from teacher forcing to using model predictions
- **Gradient Clipping**: Prevent exploding gradients in LSTM

### Evaluation Metrics
1. **BLEU Score**: Measures n-gram overlap with reference captions (BLEU-1, BLEU-4)
2. **Perplexity**: Measures how well model predicts the word distribution
3. **Qualitative**: Visual inspection of generated captions

## Expected Challenges

1. **Large Vocabulary**: Need efficient embedding and output layers
2. **Variable Caption Lengths**: Require padding and masking
3. **Image-Text Alignment**: Ensuring CNN features capture relevant details
4. **Overfitting**: Dataset may be small, need regularization (dropout)
5. **Inference Speed**: Beam search is slower but produces better captions

## Configuration Parameters (Suggested)

```python
# Data
BATCH_SIZE = 64
VAL_SPLIT = 0.2
MAX_CAPTION_LEN = 20  # Truncate longer captions

# Model
EMBED_SIZE = 256      # Word embedding dimension
HIDDEN_SIZE = 512     # LSTM hidden state size
NUM_LAYERS = 1        # LSTM layers
DROPOUT = 0.5

# Training
EPOCHS = 20
LR = 1e-3
TEACHER_FORCING_RATIO = 0.8

# Vocabulary
MIN_FREQ = 5          # Minimum word frequency
```

## Dataset Statistics (To be computed)
- Total images: ~8,000 (estimated from file count)
- Total captions: ~40,000 (5 per image)
- Vocabulary size: TBD (after building vocab)
- Average caption length: TBD
- Train/Val split: 80/20

## References & Resources
- **Paper**: "Show and Tell: A Neural Image Caption Generator" (Vinyals et al., 2015)
- **Dataset**: Flickr8k/Flickr30k style dataset
- **Pre-trained Models**: torchvision.models for CNN encoder

---

**Status**: Ready for implementation
**Date**: 2025-11-07
