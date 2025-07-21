import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import time

# --- Configuration ---
# We define some hyperparameters for our model here.
D_MODEL = 128  # Dimension of embeddings (reduced for faster CPU execution)
N_HEADS = 4    # Number of attention heads
D_HEAD = D_MODEL // N_HEADS # Dimension of each attention head
BLOCK_SIZE = 16 # The size of the blocks for local attention
TRAINING_EPOCHS = 100 # How many times to train on the data
LEARNING_RATE = 1e-3 # Step size for the optimizer

class StandardSelfAttention(nn.Module):
    """
    A standard multi-head self-attention mechanism.
    This is a highly parallelizable building block.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(context)

class ParallelHierarchicalAttention(nn.Module):
    """
    A non-recursive, parallel implementation of the hierarchical attention idea.
    This version is much faster and more efficient.
    """
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.block_size = block_size

        # We use the standard attention module as a building block
        self.local_attention = StandardSelfAttention(d_model, n_heads)
        self.summary_attention = StandardSelfAttention(d_model, n_heads)
        self.summarizer = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # 1. Pad the sequence to be a multiple of block_size for easy reshaping
        pad_len = (self.block_size - seq_len % self.block_size) % self.block_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        padded_len = x.shape[1]
        num_blocks = padded_len // self.block_size

        # 2. Perform local attention within each block in parallel
        x_reshaped = x.contiguous().view(batch_size * num_blocks, self.block_size, d_model)
        local_context = self.local_attention(x_reshaped)

        local_context = local_context.view(batch_size, padded_len, d_model)

        # 3. Create summary tokens for each block
        block_view = local_context.view(batch_size, num_blocks, self.block_size, d_model)
        summary_tokens = self.summarizer(block_view.mean(dim=2))

        # 4. Perform attention on the summary tokens (global context mixing)
        summary_context = self.summary_attention(summary_tokens)

        # 5. Distribute summary context back to the local tokens
        summary_context_expanded = summary_context.unsqueeze(2).expand(-1, -1, self.block_size, -1)
        summary_context_distributed = summary_context_expanded.contiguous().view(batch_size, padded_len, d_model)

        # 6. Combine local and summary contexts
        final_context = local_context + summary_context_distributed

        # 7. Un-pad the sequence to its original length
        return final_context[:, :seq_len, :]


class TransformerBlock(nn.Module):
    """
    A single block of the Transformer, which can use either attention mechanism.
    """
    def __init__(self, d_model, n_heads, block_size, use_hierarchical_attention=True):
        super().__init__()
        if use_hierarchical_attention:
            self.attention = ParallelHierarchicalAttention(d_model, n_heads, block_size)
        else:
            self.attention = StandardSelfAttention(d_model, n_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fed_forward = self.ffn(x)
        x = self.norm2(fed_forward + x)
        return x

# --- Main Model ---
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, n_heads, block_size, use_hierarchical):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, block_size, use_hierarchical)
            for _ in range(num_layers)
        ])

        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        batch_size, seq_len = idx.shape
        x = self.embedding(idx)
        x = x + self.pos_encoding[:, :seq_len, :]
        for layer in self.layers:
            x = layer(x)
        return self.output_head(x)

# --- Data Preparation ---
def prepare_data(text, vocab):
    tokens = re.findall(r'\b\w+\b', text.lower())
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

# --- Training Function ---
def train_model(model_name, model, train_data, val_data):
    print(f"--- Training {model_name} ---")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    train_inputs, train_targets = train_data
    val_inputs, val_targets = val_data

    start_time = time.time()
    for epoch in range(TRAINING_EPOCHS):
        # Training step
        model.train()
        optimizer.zero_grad()
        logits = model(train_inputs)
        b, t, c = logits.shape
        train_loss = loss_fn(logits.view(b * t, c), train_targets.view(b * t))
        train_loss.backward()
        optimizer.step()

        # Evaluation step
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_inputs)
                b, t, c = val_logits.shape
                val_loss = loss_fn(val_logits.view(b * t, c), val_targets.view(b * t))
                perplexity = torch.exp(val_loss)

            print(f"Epoch [{epoch+1}/{TRAINING_EPOCHS}], "
                  f"Train Loss: {train_loss.item():.4f}, "
                  f"Val Loss: {val_loss.item():.4f}, "
                  f"Perplexity: {perplexity.item():.2f}")

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    print(f"Final Val Loss for {model_name}: {val_loss.item():.4f}\n")


# --- Example Usage with Real Text ---
if __name__ == '__main__':
    story_text = """
    In a quiet village nestled between rolling hills and a whispering forest, lived a clockmaker named Alistair.
    He was a man of precision and patience, his hands weathered by years of crafting intricate gears and springs.
    His shop, filled with the gentle ticking of a hundred clocks, was a sanctuary of time itself. One day, a traveler
    arrived, carrying a strange, silent timepiece. It was made of a dark, smooth wood that seemed to absorb the light,
    and its face had no hands, only a complex spiral pattern. The traveler explained that the clock was a family heirloom,
    said to measure not hours or minutes, but opportunities. It had been silent for generations. Intrigued, Alistair
    accepted the challenge. For weeks, he studied the silent clock, his own world fading into the background. He forgot
    to wind his other clocks, and the familiar ticking in his shop slowly ceased. He discovered that the spiral was not
    a pattern, but a single, continuous groove, like a record. He realized the clock didn't need hands; it needed a needle.
    He crafted a delicate stylus from a fallen star fragment he had kept since childhood. As he placed the needle onto the
    spiral, the shop filled not with a tick, but with a soft, resonant hum. A wave of warmth spread from the timepiece,
    and Alistair felt a sudden, overwhelming clarity. He saw not the past or the future, but the present moment in all
    its infinite potential. He saw the missed chances of his life not as regrets, but as paths not taken, each beautiful
    in its own right. The traveler returned to find a changed man. Alistair returned the humming clock. 'It was never broken,'
    the clockmaker said, his voice calm. 'It was just waiting for someone to learn how to listen.' The traveler smiled,
    understanding completely. He left the village, leaving behind the clockmaker who had finally learned to measure his life
    not in the seconds he had lost, but in the richness of the now.
    """

    all_tokens = sorted(list(set(re.findall(r'\b\w+\b', story_text.lower()))))
    vocab = {token: i for i, token in enumerate(all_tokens, 2)}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    vocab_size = len(vocab)

    data_tensor = prepare_data(story_text, vocab)

    # Split data into training and validation (80% train, 20% val)
    n = int(0.8 * data_tensor.shape[1])
    train_data_raw = data_tensor[:, :n]
    val_data_raw = data_tensor[:, n:]

    train_inputs = train_data_raw[:, :-1]
    train_targets = train_data_raw[:, 1:]

    val_inputs = val_data_raw[:, :-1]
    val_targets = val_data_raw[:, 1:]

    print(f"Vocabulary Size: {vocab_size}")
    print(f"Training data shape: {train_inputs.shape}")
    print(f"Validation data shape: {val_inputs.shape}")
    print("-" * 40)

    # 1. Instantiate and Train Your NEW Parallel Hierarchical Model
    hierarchical_model = SimpleTransformer(
        vocab_size=vocab_size, num_layers=2, d_model=D_MODEL, n_heads=N_HEADS,
        block_size=BLOCK_SIZE, use_hierarchical=True
    )
    train_model("Parallel Hierarchical Attention Model", hierarchical_model, (train_inputs, train_targets), (val_inputs, val_targets))

    # 2. Instantiate and Train Standard Transformer Model
    standard_model = SimpleTransformer(
        vocab_size=vocab_size, num_layers=2, d_model=D_MODEL, n_heads=N_HEADS,
        block_size=BLOCK_SIZE, use_hierarchical=False
    )
    train_model("Standard Full-Attention Model", standard_model, (train_inputs, train_targets), (val_inputs, val_targets))
