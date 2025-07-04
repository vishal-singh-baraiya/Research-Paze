import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from tqdm import tqdm
import math
import pickle
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TextDataset(Dataset):
    """Enhanced dataset class with better tokenization and caching"""
    
    def __init__(self, text_file: str, seq_length: int = 512, vocab_size: int = 50000):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Load and preprocess text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Build vocabulary with improved tokenization
        self.vocab, self.inv_vocab = self._build_vocab(text)
        
        # Tokenize text
        self.tokens = self._tokenize(text)
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        print(f"Dataset created with {len(self.sequences)} sequences")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Total tokens: {len(self.tokens)}")
    
    def _build_vocab(self, text: str) -> Tuple[dict, dict]:
        """Build vocabulary with better handling of special tokens"""
        # Simple word-level tokenization (can be improved with BPE)
        words = text.lower().split()
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Keep most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Special tokens
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        
        # Add most frequent words
        for word, _ in sorted_words[:self.vocab_size - 4]:
            vocab[word] = len(vocab)
        
        inv_vocab = {v: k for k, v in vocab.items()}
        return vocab, inv_vocab
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text using the vocabulary"""
        words = text.lower().split()
        tokens = []
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab['<UNK>'])
        
        return tokens
    
    def _create_sequences(self) -> List[Tuple[List[int], List[int]]]:
        """Create input-target sequence pairs"""
        sequences = []
        
        for i in range(0, len(self.tokens) - self.seq_length, self.seq_length // 2):
            input_seq = self.tokens[i:i + self.seq_length]
            target_seq = self.tokens[i + 1:i + self.seq_length + 1]
            
            if len(input_seq) == self.seq_length and len(target_seq) == self.seq_length:
                sequences.append((input_seq, target_seq))
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


class MultiHeadAttention(nn.Module):
    """Optimized multi-head attention with better initialization"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Better initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Compute Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final projection
        output = self.w_o(context)
        return output


class FeedForward(nn.Module):
    """Enhanced feed-forward network with GELU activation"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Better than ReLU for transformers
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Enhanced transformer block with better normalization"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture (better than post-norm)
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class GPT1Model(nn.Module):
    """GPT-1 model with improvements"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        max_seq_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def create_causal_mask(self, seq_length: int) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_length, seq_length))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = x.size()
        
        # Create position indices
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        mask = self.create_causal_mask(seq_length).to(x.device)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Final layer norm and projection
        x = self.final_norm(x)
        logits = self.output_projection(x)
        
        return logits


class GPT1Trainer:
    """Enhanced trainer with better optimization and monitoring"""
    
    def __init__(
        self,
        model: GPT1Model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=learning_rate * 0.1
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.step = 0
    
    def warmup_lr(self, step: int):
        """Learning rate warmup"""
        if step < self.warmup_steps:
            lr_scale = step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_scale
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Single training step"""
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(inputs)
        loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update learning rate
        if self.step < self.warmup_steps:
            self.warmup_lr(self.step)
        else:
            self.scheduler.step()
        
        self.step += 1
        return loss.item()
    
    def validate(self) -> float:
        """Validation step"""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                logits = self.model(inputs)
                loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, save_path: str = 'gpt1_model.pt'):
        """Main training loop"""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        self.model.train()
        pbar = tqdm(total=self.max_steps, desc="Training")
        
        epoch = 0
        while self.step < self.max_steps:
            epoch += 1
            epoch_losses = []
            
            for batch in self.train_loader:
                if self.step >= self.max_steps:
                    break
                
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'epoch': epoch
                })
                pbar.update(1)
                
                # Validation
                if self.step % 500 == 0:
                    val_loss = self.validate()
                    self.val_losses.append(val_loss)
                    
                    if val_loss > 0:
                        print(f"\nStep {self.step}: Train Loss = {np.mean(epoch_losses):.4f}, Val Loss = {val_loss:.4f}")
            
            # Save epoch metrics
            if epoch_losses:
                self.train_losses.append(np.mean(epoch_losses))
        
        pbar.close()
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'step': self.step
        }, save_path)
        
        print(f"\nTraining completed! Model saved to {save_path}")


class GPT1Generator:
    """Enhanced text generator with better sampling strategies"""
    
    def __init__(self, model: GPT1Model, vocab: dict, inv_vocab: dict, device: str = 'cuda'):
        self.model = model.to(device)
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.device = device
        self.model.eval()
    
    def top_k_top_p_sampling(self, logits: torch.Tensor, k: int = 50, p: float = 0.95, temperature: float = 1.0) -> torch.Tensor:
        """Advanced sampling with top-k and top-p (nucleus) sampling"""
        logits = logits / temperature
        
        # Top-k sampling
        if k > 0:
            top_k = torch.topk(logits, k)[0]
            logits[logits < top_k[:, [-1]]] = -float('inf')
        
        # Top-p sampling
        if p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            # Scatter back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
        
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    def generate(
        self,
        prompt: str = "",
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1
    ) -> str:
        """Generate text with advanced sampling"""
        
        # Tokenize prompt
        if prompt:
            tokens = []
            for word in prompt.lower().split():
                if word in self.vocab:
                    tokens.append(self.vocab[word])
                else:
                    tokens.append(self.vocab['<UNK>'])
        else:
            tokens = [self.vocab.get('<BOS>', 0)]
        
        generated_tokens = tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input
                input_ids = torch.tensor([tokens], device=self.device)
                
                # Get logits
                logits = self.model(input_ids)
                next_token_logits = logits[0, -1, :].clone()
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token in set(tokens):
                        if next_token_logits[token] < 0:
                            next_token_logits[token] *= repetition_penalty
                        else:
                            next_token_logits[token] /= repetition_penalty
                
                # Sample next token
                next_token = self.top_k_top_p_sampling(
                    next_token_logits.unsqueeze(0),
                    k=top_k,
                    p=top_p,
                    temperature=temperature
                )
                
                next_token = next_token.item()
                
                # Check for end tokens
                if next_token in [self.vocab.get('<EOS>', -1), self.vocab.get('<PAD>', -1)]:
                    break
                
                tokens.append(next_token)
                generated_tokens.append(next_token)
                
                # Truncate if too long
                if len(tokens) > self.model.max_seq_length:
                    tokens = tokens[-self.model.max_seq_length:]
        
        # Convert tokens back to text
        words = []
        for token in generated_tokens:
            if token in self.inv_vocab:
                words.append(self.inv_vocab[token])
            else:
                words.append('<UNK>')
        
        return ' '.join(words)


def main():
    """Main training and inference pipeline"""
    
    # Hyperparameters
    CONFIG = {
        'vocab_size': 10000,
        'seq_length': 128,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'warmup_steps': 1000,
        'max_steps': 5000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("GPT-1 Enhanced Implementation")
    print("=" * 50)
    
    # 1. Load and prepare data
    print("\n1. Loading and preparing data...")
    try:
        dataset = TextDataset(
            'data.txt',
            seq_length=CONFIG['seq_length'],
            vocab_size=CONFIG['vocab_size']
        )
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
    except FileNotFoundError:
        print("ERROR: data.txt not found. Please create a text file named 'data.txt' in the current directory.")
        return
    
    # 2. Initialize model
    print("\n2. Initializing model...")
    model = GPT1Model(
        vocab_size=len(dataset.vocab),
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        d_ff=CONFIG['d_ff'],
        max_seq_length=CONFIG['seq_length'],
        dropout=CONFIG['dropout']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Train model
    print("\n3. Training model...")
    trainer = GPT1Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        warmup_steps=CONFIG['warmup_steps'],
        max_steps=CONFIG['max_steps'],
        device=CONFIG['device']
    )
    
    trainer.train('gpt1_enhanced.pt')
    
    # 4. Save vocabulary
    print("\n4. Saving vocabulary...")
    with open('vocab.pkl', 'wb') as f:
        pickle.dump({
            'vocab': dataset.vocab,
            'inv_vocab': dataset.inv_vocab
        }, f)
    
    # 5. Text generation
    print("\n5. Testing text generation...")
    generator = GPT1Generator(model, dataset.vocab, dataset.inv_vocab, CONFIG['device'])
    
    # Generate some sample text
    prompts = [
        "The future of artificial intelligence",
        "Once upon a time",
        "In the beginning",
        ""  # Empty prompt
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = generator.generate(
            prompt=prompt,
            max_length=50,
            temperature=0.8,
            top_k=50,
            top_p=0.95
        )
        print(f"Generated: {generated}")
        print("-" * 50)
    
    print("\nTraining completed successfully!")
    print("Files created:")
    print("- gpt1_enhanced.pt (model checkpoint)")
    print("- vocab.pkl (vocabulary)")


def load_and_generate(model_path: str = 'gpt1_enhanced.pt', vocab_path: str = 'vocab.pkl'):
    """Load trained model and generate text"""
    
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    
    vocab = vocab_data['vocab']
    inv_vocab = vocab_data['inv_vocab']
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize model (you may need to adjust these parameters)
    model = GPT1Model(
        vocab_size=len(vocab),
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_length=128,
        dropout=0.1
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create generator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = GPT1Generator(model, vocab, inv_vocab, device)
    
    return generator


if __name__ == "__main__":
    main()
