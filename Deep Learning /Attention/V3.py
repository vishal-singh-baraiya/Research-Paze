
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import time
import os
import gc
import logging
from typing import Optional, List, Tuple
from torch.utils.data import Dataset, DataLoader

# --- Configuration ---
D_MODEL = 64  # Model dimension
N_HEADS = 4   # Number of attention heads
D_HEAD = D_MODEL // N_HEADS
BLOCK_SIZE = 32  # Block size for hierarchical attention
BTREE_ORDER = 5  # B+ tree order
TRAINING_EPOCHS = 200
LEARNING_RATE = 5e-4
DROPOUT = 0.1
PATIENCE = 10
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 4
MAX_CACHE_SIZE = 50

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Custom Dataset for loading text sequences"""
    def __init__(self, indices, seq_len=64):
        self.indices = indices
        self.seq_len = seq_len
        self.sequences = []
        for i in range(0, len(indices) - seq_len, seq_len):
            self.sequences.append(indices[i:i + seq_len])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]

class BTreeNode:
    """B+ Tree node for efficient attention pattern management"""
    def __init__(self, order: int, is_leaf: bool = False, device: str = 'cuda'):
        self.order = order
        self.is_leaf = is_leaf
        self.keys = []
        self.values = []
        self.children = [] if not is_leaf else []
        self.next = None
        self.attention_cache = {}
        self.device = device

    def is_full(self) -> bool:
        return len(self.keys) >= self.order - 1

    def insert_key(self, key: int, value: torch.Tensor):
        i = 0
        while i < len(self.keys) and self.keys[i] < key:
            i += 1
        self.keys.insert(i, key)
        self.values.insert(i, value.to(self.device))

    def search(self, key: int) -> Optional[torch.Tensor]:
        if key in self.attention_cache:
            return self.attention_cache[key]
        for i, k in enumerate(self.keys):
            if k == key:
                return self.values[i]
        return None

    def range_search(self, start: int, end: int) -> List[torch.Tensor]:
        results = []
        for i, key in enumerate(self.keys):
            if start <= key <= end:
                results.append(self.values[i])
        return results

class BTreeAttentionIndex:
    """B+ Tree structure for managing attention patterns"""
    def __init__(self, order: int = 5, device: str = 'cuda'):
        self.order = order
        self.root = BTreeNode(order, is_leaf=True, device=device)
        self.height = 1
        self.device = device
        self.cache_count = 0
        self.max_cache_size = MAX_CACHE_SIZE

    def insert(self, key: int, value: torch.Tensor):
        if self.cache_count >= self.max_cache_size:
            logger.debug(f"Cache limit reached ({self.max_cache_size}), skipping insertion for key {key}")
            return
        if self.root.is_full():
            logger.debug(f"Root is full, splitting. Current keys: {self.root.keys}")
            new_root = BTreeNode(self.order, is_leaf=False, device=self.device)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
            self.height += 1
        self._insert_non_full(self.root, key, value)
        self.cache_count += 1
        logger.debug(f"Inserted key {key}, cache count: {self.cache_count}")

    def _insert_non_full(self, node: BTreeNode, key: int, value: torch.Tensor):
        if node.is_leaf:
            node.insert_key(key, value)
        else:
            i = len(node.keys)
            while i > 0 and key < node.keys[i-1]:
                i -= 1
            if node.children[i].is_full():
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key, value)

    def _split_child(self, parent: BTreeNode, index: int):
        full_child = parent.children[index]
        new_child = BTreeNode(self.order, full_child.is_leaf, device=self.device)
        mid = (self.order - 1) // 2
        new_child.keys = full_child.keys[mid + 1:]
        new_child.values = full_child.values[mid + 1:]
        full_child.keys = full_child.keys[:mid]
        full_child.values = full_child.values[:mid]
        if not full_child.is_leaf:
            new_child.children = full_child.children[mid + 1:]
            full_child.children = full_child.children[:mid + 1]
        else:
            new_child.next = full_child.next
            full_child.next = new_child
        parent.children.insert(index + 1, new_child)
        parent.keys.insert(index, new_child.keys[0])
        logger.debug(f"Splitting node at index {index}. Full child keys: {full_child.keys}, New child keys: {new_child.keys}")

    def range_query(self, start: int, end: int) -> List[torch.Tensor]:
        def _range_search(node: BTreeNode, start: int, end: int) -> List[torch.Tensor]:
            results = []
            if node.is_leaf:
                results.extend(node.range_search(start, end))
            else:
                for i, key in enumerate(node.keys):
                    if start <= key:
                        results.extend(_range_search(node.children[i], start, end))
                    if key > end:
                        break
                if (not node.keys or end > node.keys[-1]) and len(node.children) > len(node.keys):
                    results.extend(_range_search(node.children[-1], start, end))
            return results
        return _range_search(self.root, start, end)

class OptimizedParallelHierarchicalAttention(nn.Module):
    """Enhanced PHA with B+ Tree indexing"""
    def __init__(self, d_model, n_heads, block_size, btree_order=5, device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.block_size = block_size
        self.btree_order = btree_order
        self.device = device
        self.local_attention = StandardSelfAttention(d_model, n_heads, use_position_bias=True, block_size=block_size).to(device)
        self.summary_attention = StandardSelfAttention(d_model, n_heads).to(device)
        self.summarizer = nn.Linear(d_model, d_model).to(device)
        self.dropout = nn.Dropout(DROPOUT).to(device)
        self.attention_index = BTreeAttentionIndex(btree_order, device=device)
        self.pattern_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_cached_attention_pattern(self, seq_len: int, batch_size: int) -> Optional[torch.Tensor]:
        padded_len = seq_len + (self.block_size - seq_len % self.block_size) % self.block_size
        cache_key = (padded_len, batch_size)
        if cache_key in self.pattern_cache:
            self.cache_hits += 1
            return self.pattern_cache[cache_key].to(self.device)
        self.cache_misses += 1
        return None

    def _cache_attention_pattern(self, seq_len: int, batch_size: int, pattern: torch.Tensor):
        padded_len = seq_len + (self.block_size - seq_len % self.block_size) % self.block_size
        cache_key = (padded_len, batch_size)
        if len(self.pattern_cache) < MAX_CACHE_SIZE:
            self.pattern_cache[cache_key] = pattern.detach().clone().to(self.device)
            self.attention_index.insert(padded_len, pattern.mean(dim=(0, 1)))

    def _compute_efficient_local_attention(self, x_reshaped: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_blocks, block_len, d_model = x_reshaped.shape
        batch_size = batch_blocks // (x_reshaped.shape[1] // self.block_size)
        cached_pattern = self._get_cached_attention_pattern(block_len, batch_size)
        if cached_pattern is not None:
            context, _ = self.local_attention(x_reshaped, cached_pattern=cached_pattern, return_weights=True)
            return context, cached_pattern
        context, attention_weights = self.local_attention(x_reshaped, return_weights=True)
        self._cache_attention_pattern(block_len, batch_size, attention_weights)
        return context, attention_weights

    def _hierarchical_summary_with_btree(self, block_summaries: torch.Tensor) -> torch.Tensor:
        batch_size, num_blocks, d_model = block_summaries.shape
        if num_blocks > 1:
            relevant_patterns = self.attention_index.range_query(
                max(0, num_blocks - self.btree_order), num_blocks + self.btree_order
            )
            if relevant_patterns:
                pattern_prior = torch.stack(relevant_patterns).mean(dim=0).to(self.device)
                pattern_prior = pattern_prior.unsqueeze(0).unsqueeze(0)
                return self.summary_attention(block_summaries, bias=pattern_prior)
        return self.summary_attention(block_summaries)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        pad_len = (self.block_size - seq_len % self.block_size) % self.block_size
        if pad_len > 0:
            padding = torch.zeros(batch_size, pad_len, d_model, device=self.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        padded_len = x.shape[1]
        num_blocks = padded_len // self.block_size
        x_reshaped = x.view(batch_size * num_blocks, self.block_size, d_model)
        local_context, _ = self._compute_efficient_local_attention(x_reshaped)
        local_context = local_context.view(batch_size, padded_len, d_model)
        local_context = self.dropout(local_context)
        block_view = local_context.view(batch_size, num_blocks, self.block_size, d_model)
        weights = torch.softmax(torch.randn(self.block_size, device=self.device), dim=0)
        summary_tokens = self.summarizer((block_view * weights.view(1, 1, -1, 1)).sum(dim=2))
        summary_context = self._hierarchical_summary_with_btree(summary_tokens)
        summary_context = self.dropout(summary_context)
        summary_context_expanded = summary_context.unsqueeze(2).expand(-1, -1, self.block_size, -1)
        summary_context_distributed = summary_context_expanded.reshape(batch_size, padded_len, d_model)
        final_context = local_context + summary_context_distributed
        return final_context[:, :seq_len, :]

    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'btree_height': self.attention_index.height
        }

class StandardSelfAttention(nn.Module):
    """Standard multi-head self-attention mechanism"""
    def __init__(self, d_model, n_heads, use_position_bias=False, block_size=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(DROPOUT)
        self.use_position_bias = use_position_bias
        if use_position_bias:
            if block_size is None:
                raise ValueError("block_size must be provided if use_position_bias is True")
            self.block_size = block_size
            self.position_bias = nn.Parameter(torch.randn(1, n_heads, block_size, block_size) * 0.1)

    def forward(self, x, mask=None, cached_pattern=None, return_weights=False):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if self.use_position_bias and seq_len <= self.block_size:
            bias = self.position_bias[:, :, :seq_len, :seq_len]
            scores = scores + bias
        if cached_pattern is not None:
            scores = scores + cached_pattern
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        context = self.out_proj(context)
        if return_weights:
            return context, attention_weights
        return context

class EnhancedTransformerBlock(nn.Module):
    """Enhanced transformer block with B+ tree optimization"""
    def __init__(self, d_model, n_heads, block_size, device='cuda'):
        super().__init__()
        self.attention = OptimizedParallelHierarchicalAttention(
            d_model, n_heads, block_size, BTREE_ORDER, device=device
        )
        self.norm1 = nn.LayerNorm(d_model).to(device)
        self.norm2 = nn.LayerNorm(d_model).to(device)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(d_model * 2, d_model)
        ).to(device)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        fed_forward = self.ffn(x)
        x = self.norm2(fed_forward + x)
        return x

class OptimizedSimpleTransformer(nn.Module):
    """Enhanced transformer with B+ tree optimizations"""
    def __init__(self, vocab_size, num_layers, d_model, n_heads, block_size, device='cuda'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model).to(device)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.1).to(device)
        self.layers = nn.ModuleList([
            EnhancedTransformerBlock(d_model, n_heads, block_size, device=device)
            for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(d_model, vocab_size).to(device)
        self.device = device
        self.vocab_size = vocab_size

    def forward(self, idx):
        batch_size, seq_len = idx.shape
        x = self.embedding(idx)
        x = x + self.pos_encoding[:, :seq_len, :]
        for layer in self.layers:
            x = layer(x)
        return self.output_head(x)

    def get_model_stats(self):
        stats = {}
        for i, layer in enumerate(self.layers):
            stats[f'layer_{i}'] = layer.attention.get_cache_stats()
        return stats

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text sequences given an input index sequence.
        
        Args:
            idx: Input indices (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for controlling randomness
            top_k: If specified, only consider top-k logits for sampling
        
        Returns:
            Generated indices
        """
        self.eval()
        generated = idx.clone()
        for _ in range(max_new_tokens):
            idx_cond = generated[:, -self.pos_encoding.size(1):]
            with torch.no_grad():
                logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                top_k = min(top_k, self.vocab_size)  # Ensure top_k does not exceed vocab size
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated

def prepare_data(text, vocab, seq_len=64):
    """Prepare data with reduced overlap"""
    tokens = re.findall(r'\b\w+\b', text.lower())
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    return torch.tensor(indices, dtype=torch.long)

def train_model(model_name, model, train_loader, val_loader, device='cuda'):
    print(f"--- Training {model_name} ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, TRAINING_EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    patience_counter = 0

    start_time = time.time()
    for epoch in range(TRAINING_EPOCHS):
        model.train()
        total_train_loss = 0
        num_batches = 0
        optimizer.zero_grad()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            b, t, c = logits.shape
            loss = loss_fn(logits.view(b * t, c), targets.view(b * t))
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            total_train_loss += loss.item() * GRAD_ACCUM_STEPS
            num_batches += 1
            torch.cuda.empty_cache()
        train_loss = total_train_loss / num_batches

        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                b, t, c = logits.shape
                val_loss = loss_fn(logits.view(b * t, c), targets.view(b * t))
                total_val_loss += val_loss.item()
                num_val_batches += 1
        val_loss = total_val_loss / num_val_batches
        perplexity = torch.exp(torch.tensor(val_loss))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{TRAINING_EPOCHS}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Perplexity: {perplexity.item():.2f}")
            stats = model.get_model_stats()
            avg_hit_rate = sum(s['hit_rate'] for s in stats.values()) / len(stats)
            print(f"Avg Cache Hit Rate: {avg_hit_rate:.3f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
        scheduler.step()
        torch.cuda.empty_cache()
        gc.collect()

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    print(f"Final Val Loss for {model_name}: {best_val_loss:.4f}\n")

def decode_indices(indices, reverse_vocab):
    """Convert indices back to text using reverse vocabulary"""
    return ' '.join([reverse_vocab.get(idx.item(), '<unk>') for idx in indices])

if __name__ == '__main__':
    device = 'cuda'
    print(f"Using device: {device} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")

    # Read data from data.txt
    data_file = 'data.txt'
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Error: {data_file} not found in the current directory.")
    with open(data_file, 'r', encoding='utf-8') as f:
        story_text = f.read().strip()
    if not story_text:
        raise ValueError(f"Error: {data_file} is empty.")

    all_tokens = sorted(list(set(re.findall(r'\b\w+\b', story_text.lower()))))
    vocab = {token: i for i, token in enumerate(all_tokens, 2)}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    reverse_vocab = {i: token for token, i in vocab.items()}
    vocab_size = len(vocab)
    seq_len = 64
    indices = prepare_data(story_text, vocab, seq_len)
    dataset = TextDataset(indices, seq_len)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Training dataset size: {len(train_dataset)} sequences")
    print(f"Validation dataset size: {len(val_dataset)} sequences")
    print("-" * 50)

    # Train B+ Tree Optimized Model
    model = OptimizedSimpleTransformer(
        vocab_size=vocab_size, num_layers=2, d_model=D_MODEL, n_heads=N_HEADS,
        block_size=BLOCK_SIZE, device=device
    ).to(device)
    train_model("B+ Tree Optimized PHA Model", model, train_loader, val_loader, device)

    print("=== Final Model Statistics ===")
    stats = model.get_model_stats()
    for layer_name, stat in stats.items():
        print(f"{layer_name}: Hit Rate = {stat['hit_rate']:.3f}, "
              f"B+ Tree Height = {stat['btree_height']}")

    # Inference
    print("\n=== Inference ===")
    sample_idx, _ = val_dataset[0]
    prompt = sample_idx[:10].unsqueeze(0).to(device)
    print("Prompt:", decode_indices(prompt[0], reverse_vocab))
    
    print("\nGenerating with B+ Tree Optimized Model:")
    generated = model.generate(prompt, max_new_tokens=50, temperature=0.7, top_k=50)  # Adjusted parameters
    print("Generated:", decode_indices(generated[0], reverse_vocab))

    torch.cuda.empty_cache()
    gc.collect()
