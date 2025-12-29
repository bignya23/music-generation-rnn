# 🎵 Music Generation with RNNs

Generate original Irish folk music using deep learning! This project implements a character-level Recurrent Neural Network (LSTM) that learns patterns from thousands of songs in ABC notation and creates entirely new musical compositions.

## 📋 Overview

This project demonstrates how sequence models can learn the structure of music and generate creative outputs. By training on a dataset of Irish folk songs, the model learns:

- Musical patterns and structures
- ABC notation syntax and conventions
- Rhythm, melody, and harmonic relationships
- Song metadata (titles, keys, tempo)

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
torch
numpy
comet-ml
mitdeeplearning
scipy
tqdm
```

### Installation

```bash
# Clone or download the repository
cd introtodeeplearning/lab1

# Install dependencies
pip install torch numpy comet-ml scipy tqdm mitdeeplearning
```

### Running the Notebooks

**PyTorch Version:**

```bash
jupyter notebook PT_Part2_Music_Generation.ipynb
```

**TensorFlow Version:**

```bash
jupyter notebook TF_Part2_Music_Generation.ipynb
```

## 🎼 What is ABC Notation?

ABC notation is a text-based music notation system that uses ASCII characters to represent musical notes, rhythms, and structures. Example:

```
X:1
T:Speed the Plough
M:4/4
L:1/8
K:G
|:GABc dedB|dedB dedB|c2ec B2dB|c2A2 A2BA|
```

## 🏗️ Model Architecture

The model uses an LSTM-based architecture:

1. **Embedding Layer**: Converts character indices to dense vectors (256-dim)
2. **LSTM Layer**: Processes sequences and maintains temporal state (1024 hidden units)
3. **Dense Output Layer**: Projects to vocabulary size with softmax

```
Input (batch, seq_len)
  → Embedding (batch, seq_len, 256)
  → LSTM (batch, seq_len, 1024)
  → Linear (batch, seq_len, vocab_size)
  → Softmax → Predicted characters
```

## 🎯 Key Features

- **Character-level prediction**: Predicts the next character given a sequence
- **Temperature sampling**: Control creativity vs. coherence in generation
- **Experiment tracking**: Integration with Comet ML for monitoring training
- **Checkpoint saving**: Save and resume training at any point
- **Audio playback**: Convert generated ABC notation to playable audio files

## 📊 Training

### Hyperparameters

```python
num_training_iterations = 3000
batch_size = 8
seq_length = 100
learning_rate = 5e-3
embedding_dim = 256
hidden_size = 1024
```

### Training Process

1. Load and vectorize the Irish folk song dataset
2. Create training batches with sequence length = 100
3. Train LSTM to predict next character at each time step
4. Monitor loss convergence (expect ~1.5-2.0 final loss)
5. Save model checkpoints every 100 iterations

## 🎨 Music Generation

Once trained, generate new music by:

1. Providing a seed string (e.g., "X:1\n")
2. Iteratively sampling from the model's predictions
3. Converting ABC output to audio format
4. Playing back the generated song

```python
generated_text = generate_text(model, start_string="X:1\n", generation_length=1000)
```

## 📁 Project Structure

```
lab1/
├── PT_Part2_Music_Generation.ipynb    # PyTorch implementation
├── TF_Part2_Music_Generation.ipynb    # TensorFlow implementation
├── solutions/
│   ├── PT_Part2_Music_Generation_Solution.ipynb
│   └── TF_Part2_Music_Generation_Solution.ipynb
├── training_checkpoints/              # Saved model weights
└── README.md
```

## 🏆 Competition & Submissions

Submit your best-generated songs for prizes! Requirements:

- Recording of your song (.wav/.mp3)
- Jupyter notebook with code
- Architecture description and hyperparameters

**Submission format:** `[FirstName]_[LastName]_RNNMusic.zip`

[Submit here](https://www.dropbox.com/request/U8nND6enGjirujVZKX1n)

## 🔧 Tips for Better Results

1. **Train longer**: More iterations = better musical structure
2. **Adjust sequence length**: Longer sequences capture more context
3. **Tune learning rate**: Balance between speed and stability
4. **Experiment with hidden size**: Larger = more capacity but slower
5. **Try different start strings**: Seed text influences generation style

## 📈 Expected Results

- **After 500 iterations**: Basic ABC syntax, some valid notes
- **After 1500 iterations**: Recognizable musical patterns
- **After 3000+ iterations**: Coherent melodies and song structures

## 🛠️ Troubleshooting

**LSTM dimension mismatch error:**

- Ensure `batch_first=True` in LSTM initialization
- Check input shapes: `(batch_size, seq_length)` expected

**Loss not decreasing:**

- Lower learning rate (try 1e-3 or 1e-4)
- Verify data preprocessing is correct
- Check for gradient clipping needs

**Generated music is nonsensical:**

- Train for more iterations
- Verify model architecture is correct
- Check loss convergence during training

## 📚 Resources

- [ABC Notation Guide](https://en.wikipedia.org/wiki/ABC_notation)
- [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [MIT Deep Learning Course](http://introtodeeplearning.com)
- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

## 🎓 Learning Objectives

By completing this project, you will:

- Understand how RNNs/LSTMs process sequential data
- Learn to implement character-level language models
- Practice hyperparameter tuning for deep learning
- Experience creative applications of AI in music
- Gain hands-on experience with PyTorch/TensorFlow

## 🤝 Contributing

This is an educational project from MIT's Introduction to Deep Learning course. Feel free to experiment and share your generated music!

## 📜 License

MIT License - See course materials for details

## 🙏 Acknowledgments

- MIT Introduction to Deep Learning (6.S191)
- Irish Folk Song Dataset
- Comet ML for experiment tracking

---

**Made with ❤️ and 🎵 by MIT Deep Learning**

_Have fun and happy music making!_ 🎶
