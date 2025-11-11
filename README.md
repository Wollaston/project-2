# Project 2 (Due: 11/14/2025 11:59PM)
This is the second project in this semester, and you're going to implement a sequence-to-sequence Transformer model to recover original sentences from preprocessed text. The goal of this project is to deepen your understanding of attention mechanisms, encoder-decoder architectures, and autoregressive text generation.

## Starter code structure
The project skeleton has the following structure:

```
├── dataset.py
├── main.py
├── model.py
├── pyproject.toml
├── README.md
├── tokenizer.py
└── uv.lock
```

## Setup

### Prerequisites
You must make sure that the following software are installed:
* `uv`
* `git`

### Clone the starter code
Run the following commands to clone starter code and prepare for your own branch to implement the project:

```bash
git clone <repo>                      # Clone starter code to local

cd <project_dir>                      # Change to project directory path
git checkout -b <your_last_name>_pa2  # Create and switch to your feature branch
```

### Download dataset
In Moodle, download the `.zip` file named `project_2_data` to your local machine and unzip it to the project
directory. The dataset contains three files:
* `train.json` - Training set
* `dev.json` - Development/validation set
* `test.json` - Test set

Each file contains sequence pairs in JSON format where:
* `src`: preprocessed sentence (lemmatized with removed determiners, conjunctions, and prepositions)
* `tgt`: original sentence in lowercase

> [!IMPORTANT]
> Make sure you do not push dataset to GitHub!


### Python environment
You're given absolutely necessary external libraries as follows:
* `numpy`
* `torch ~= 2.6.0`
* `tqdm`
* `click`
* `nltk`

Normally you don't need other external libraries/packages, so make sure you really need them before you add any.

When the first time you enter the project directory, sync and install dependencies:
```bash
uv sync
```

To add a library/package:
```bash
uv add <pkg_name>
```
To remove a library/package:
```bash
uv remove <pkg_name>
```

## Instructions
The project has three components to complete:
1. Understand the implementations of the tokenizer and the model
2. Implementations of the dataset and the training pipeline
3. Experiments
4. A writeup for your findings, which follows an academic writing convention

### Task Description
Your task is to build a sequence-to-sequence model that recovers original sentences from preprocessed versions.

**Example:**
* **Source (preprocessed)**: `with scarcely semi - colon after his hearty thanks , little man begin his recital :`
* **Target (original)**: `with scarcely a semi - colon after his hearty thanks , the little man began his recital :`

The preprocessing removes determiners (DT), conjunctions (CC), and prepositions (TO), and lemmatizes the remaining words. Your model needs to learn to reverse this process.

### Implementations

#### Overview

This project follows a modular design with four main components:

**PROVIDED FILES** (Read and understand these):
* **`tokenizer.py`**: Vocabulary building and token encoding/decoding
* **`model.py`**: Complete the Transformer encoder-decoder architecture

**FILES TO IMPLEMENT** (Your work):
* **`dataset.py`**: PyTorch `Dataset` class for loading and preprocessing sequence pairs
* **`main.py`**: Training loop, evaluation, and experiment orchestration

**How They Fit Together**:
1. **Tokenizer** builds vocabulary from training data and converts text to IDs and vice versa
2. **Dataset** loads JSON files, uses tokenizer to encode sequences, prepares batches
3. **Model** processes encoded sequences and generates predictions
4. **Main script** coordinates everything: creates datasets, trains model, evaluates results

**Execution**:
Run the main script with three command-line arguments specifying paths to the train, dev, and test JSON files. For example, use the uv run command followed by main.py and the three file paths.

> [!NOTE]
> You're very welcome to add CLI options for configuration files (in yaml or json format) to make hyperparameter tuning easier.

---

#### Component Details

The following sections provide detailed descriptions of each module. **Read the PROVIDED sections carefully** to understand the tools at your disposal. **Implement the TO IMPLEMENT sections** following the specifications.

### Tokenizer (`tokenizer.py`) - PROVIDED
You're given a complete word-level tokenizer implementation. **Read and understand this module carefully** as you'll use it extensively in your dataset and main script implementations.

#### Key Classes:

**`Pair`**:
* A dataclass representing a source-target sentence pair
* Attributes: `src` (preprocessed sentence), `tgt` (original sentence), and `sent` (concatenation of both)
* Automatically validates that inputs are strings
* Example: `Pair(src="little man begin recital", tgt="the little man began his recital")`

**`TokenizerConfig`**:
* Configuration dataclass for tokenizer settings
* `special_tokens`: List of special tokens (default: `['<bos>', '<eos>', '<pad>', '<unk>']`)
* `min_freq`: Minimum frequency threshold for including tokens in vocabulary (default: 2)
* Tokens appearing fewer than `min_freq` times will be excluded from vocabulary and mapped to `<unk>`

**`Tokenizer`**:
* Main tokenizer class with vocabulary management and encoding/decoding capabilities

**Core Methods You'll Use:**

* **`from_file(fpath)`**: Builds vocabulary from a JSON data file
  * Reads all sentence pairs and tokenizes them
  * Counts token frequencies using Python's `Counter`
  * Filters out tokens below `min_freq` threshold
  * Adds special tokens to vocabulary
  * Creates bidirectional mappings: `word2idx` (str→int) and `idx2word` (int→str)
  * **Usage**: Call this once on your training data before creating datasets

* **`tokenize(sentence)`**: Static method for basic tokenization
  * Converts to lowercase, strips whitespace, splits on spaces
  * Example: `"The Cat sat" → ["the", "cat", "sat"]`
  * **Note**: This is a simple word-level tokenizer (not subword/BPE)

* **`encode(tokens)`**: Converts list of tokens to list of integer IDs
  * Maps each token to its vocabulary index
  * Unknown tokens automatically mapped to `<unk>` ID
  * Raises error if tokenizer hasn't been built yet
  * Example: `["the", "cat"] → [5, 142]`

* **`decode(ids)`**: Converts list of integer IDs back to tokens
  * Inverse operation of `encode()`
  * Example: `[5, 142] → ["the", "cat"]`

**Properties You'll Need:**
* `bos_id`: ID of `<bos>` (beginning-of-sequence) token
* `eos_id`: ID of `<eos>` (end-of-sequence) token
* `pad_id`: ID of `<pad>` (padding) token
* `unk_id`: ID of `<unk>` (unknown) token
* `src_vocab`: Returns the word→ID mapping dictionary
* `tgt_vocab`: Returns the ID→word mapping dictionary

**Special Tokens Explained:**
* `<bos>`: Marks the beginning of a sequence (prepended to sequences)
* `<eos>`: Marks the end of a sequence (appended to sequences)
* `<pad>`: Used to pad shorter sequences to a fixed length in batches
* `<unk>`: Replaces any out-of-vocabulary words during encoding

**Important Implementation Details:**
* The tokenizer builds a **shared vocabulary** from both source and target sequences
* Vocabulary is built only from the training data (via `from_file()`)
* The same tokenizer instance must be reused for dev and test sets
* Tokens below `min_freq` threshold are excluded to reduce vocabulary size
* All encoding/decoding operations require the tokenizer to be built first (`_is_built` flag)

### Dataset (`dataset.py`) - TO IMPLEMENT
You need to implement a PyTorch Dataset class for loading and preprocessing sequence pairs. This class will be used to create DataLoader objects for training, validation, and testing.

#### Required Class: `SeqPairDataset(Dataset)`

Your dataset class should inherit from `torch.utils.data.Dataset` and implement the standard PyTorch dataset interface.

**Constructor: `__init__(self, data_file, tokenizer, max_src_len, max_tgt_len)`**

Parameters:
* `data_file`: Path to JSON file containing sequence pairs
* `tokenizer`: A built `Tokenizer` instance (vocabulary already created)
* `max_src_len`: Maximum length for source sequences (including special tokens)
* `max_tgt_len`: Maximum length for target sequences (including special tokens)

Your constructor should:
1. **Load the JSON data** from `data_file`
   * Each entry has `"src"` (preprocessed) and `"tgt"` (original) fields
   * Use `json.load()` to read the file

2. **Process each sequence pair**:
   * Create a `Pair` object (from tokenizer module) to validate the data
   * Tokenize source and target using `Tokenizer.tokenize()`
   * Encode tokens to IDs using `tokenizer.encode()`
   * Add special tokens: prepend `<bos>` and append `<eos>` to both sequences
   * Trim sequences if they exceed `max_len - 2` (to leave room for special tokens)
   * Pad sequences to exact `max_src_len` and `max_tgt_len` using `<pad>` token

3. **Create training pairs for teacher forcing**:
   * **Encoder input**: Full padded source sequence with BOS and EOS tokens
   * **Decoder input**: Target sequence shifted right by removing the last token. This should start with BOS and end one position before the last token.
   * **Labels**: Target sequence shifted left by removing the first token. This should start one position after BOS and end with EOS.
   * Example: If target IDs are BOS, "the", "cat", "sat", EOS, PAD, PAD, then decoder input is BOS, "the", "cat", "sat", EOS, PAD (last removed), and labels are "the", "cat", "sat", EOS, PAD, PAD (first removed).

4. **Store processed samples** in a list for later retrieval

**Required Methods:**

* **`__len__(self)`**: Return the number of samples in the dataset

* **`__getitem__(self, idx)`**: Return a single sample as a tuple
  * Returns: `(encoder_input_ids, decoder_input_ids, label_ids)`
  * All three should be `torch.LongTensor` objects

**Helper Methods (Recommended):**

Consider implementing these private methods to keep your code clean:
* `_add_specials_and_trim(token_ids, max_len)`: Add `<bos>` and `<eos>`, trim if needed
* `_pad(token_ids, max_len)`: Pad sequence to target length with `<pad>` tokens

**Important Details:**
* The decoder input and labels enable **teacher forcing** during training
  * The model sees the correct previous tokens (not its own predictions)
  * Labels are shifted by one position to predict the next token
* Padding tokens in labels should be ignored during loss computation
  * PyTorch's `CrossEntropyLoss` handles this with `ignore_index=pad_id`
* Both source and target use the same tokenizer vocabulary
* All sequences in a batch must have the same length (that's why we pad)

### Model (`model.py`) - PROVIDED
You're given a complete Transformer encoder-decoder architecture. **Understanding this architecture is essential** for debugging, running experiments, and writing your report. Read through the implementation to see how the components fit together.

#### Utility Functions (Masking):

**`make_padding_mask(seq, pad_idx=0)`**:
* Creates a boolean mask indicating which positions are valid (not padding)
* Input: `seq` of shape `(batch, seq_len)` containing token IDs
* Output: Boolean tensor of shape `(batch, seq_len)` where `True` = valid token, `False` = padding
* Used by encoder to ignore padding in source sequences

**`make_causal_mask(sz)`**:
* Creates a causal (triangular) mask to prevent attending to future positions
* Input: `sz` (sequence length)
* Output: Boolean tensor of shape `(sz, sz)` where position `(i, j)` is `True` if `j <= i`
* Ensures autoregressive property: token at position `i` can only attend to positions `0` through `i`

**`make_decoder_self_mask(tgt_seq, pad_idx=0)`**:
* Combines causal mask with padding mask for decoder self-attention
* Ensures decoder cannot attend to future tokens OR padding tokens
* Output: Shape `(batch, 1, seq_len, seq_len)` ready for broadcasting

**Masking Convention**:
* `True` = valid position (can attend)
* `False` = masked position (cannot attend)
* In attention computation, `False` positions are filled with `-inf` before softmax

#### Core Architecture Components:

**1. `PosEncoding` (Positional Encoding)**:
* Adds positional information to token embeddings using sinusoidal functions
* **Why needed**: Transformers have no inherent notion of token order (unlike RNNs)
* Uses fixed sinusoidal patterns (not learned parameters)
* Formula: Uses sine for even indices and cosine for odd indices, with frequencies decreasing exponentially with position dimension
* Registered as a buffer (saved with model but not trainable)
* **Forward**: Adds positional encodings to input embeddings

**2. `MHA` (Multi-Head Attention)**:
* Core attention mechanism of Transformers
* **Parameters**:
  * d_model: Model dimension (must be divisible by num_heads)
  * num_heads: Number of attention heads
  * dropout: Dropout rate applied to attention weights
* **Components**:
  * Linear projection layers for queries, keys, and values
  * Output projection layer after concatenating heads
* **Forward method**: Takes queries, keys, values, and optional mask
  * Projects Q, K, V and splits into multiple heads
  * Computes scaled dot-product attention: applies softmax to the scaled product of queries and keys transpose, then multiplies by values
  * Applies mask (if provided) by setting masked positions to negative infinity
  * Concatenates heads and applies output projection
  * Returns both output and attention weights

**3. `FFN` (Feedforward Network)**:
* Position-wise feedforward network applied after attention
* Two linear layers with ReLU activation between them
* **Parameters**: d_model for input/output dimensions, d_ff for hidden layer size, dropout
* Applied identically to each position independently

**4. `EncoderLayer`**:
* Single layer of the Transformer encoder
* **Architecture**:
  1. Multi-head self-attention with residual connection and layer normalization
  2. Feedforward network with residual connection and layer normalization
* **Forward method**: Takes input x and optional mask
  * Self-attention: attends to all positions in encoder input
  * Returns both output and attention weights

**5. `Encoder`**:
* Full encoder stack for processing source sequences
* **Components**:
  * Token embedding layer (converts token IDs to dense vectors)
  * Positional encoding
  * Stack of encoder layers (configurable number)
  * Final layer normalization
* **Forward method**: Takes input_ids as input
  * Embeds tokens and adds positional encoding
  * Creates padding mask from input
  * Passes through all encoder layers
  * Returns encoder output, padding mask, and list of attention weights

**6. `DecoderLayer`**:
* Single layer of the Transformer decoder
* **Architecture**:
  1. Masked multi-head self-attention (causal) with residual connection and layer normalization
  2. Cross-attention to encoder outputs with residual connection and layer normalization
  3. Feedforward network with residual connection and layer normalization
* **Forward method**: Takes input x, encoder outputs, self-attention mask, and cross-attention mask
  * Self-attention: attends only to previous positions in target (causal masking)
  * Cross-attention: attends to all positions in encoder output
  * Returns output and a tuple of self-attention weights and cross-attention weights

**7. `Decoder`**:
* Full decoder stack for generating target sequences
* **Components**:
  * Token embedding layer
  * Positional encoding
  * Stack of decoder layers (configurable number)
  * Final layer normalization
* **Forward method**: Takes target IDs, encoder outputs, and encoder mask
  * Creates combined causal and padding mask for decoder self-attention
  * Embeds tokens and adds positional encoding
  * Passes through all decoder layers with cross-attention to encoder
  * Returns decoder output and list of attention weights

**8. `EncoderDecoder`** (Main Model):
* Complete sequence-to-sequence Transformer model
* **Constructor Parameters**:
  * src_vocab_size and tgt_vocab_size: Source and target vocabulary sizes
  * d_model: Model dimension (default: 128)
  * num_heads: Number of attention heads (default: 4)
  * d_ff: Feedforward hidden dimension (default: 512)
  * num_enc_layers and num_dec_layers: Number of encoder and decoder layers (default: 2 each)
  * max_len: Maximum sequence length (default: 50)
  * dropout: Dropout rate (default: 0.1)
  * pad_idx: Padding token ID (default: 0)

* **Methods**:
  * **forward method**: Takes source IDs and target IDs for training forward pass
    * Encodes source sequence using the encoder
    * Decodes target sequence with teacher forcing using the decoder
    * Projects decoder output to vocabulary size
    * Returns logits with shape batch size by target length by vocabulary size

  * **generate method**: Takes source IDs, BOS ID, EOS ID, max length, strategy, and beam width for inference
    * Generates sequences autoregressively without teacher forcing
    * Supports two strategies: greedy decoding (select highest probability token) and beam search (maintain top-k sequences)
    * Returns list of generated token ID sequences

  * **_greedy_decode (private method)**: Implements greedy decoding
    * Iteratively generates one token at a time
    * Selects argmax of output distribution at each step
    * Stops when EOS token is generated or max length is reached

  * **_beam_search (private method)**: Implements beam search decoding
    * Maintains beam_width most probable sequences
    * Expands each beam with top-k next tokens
    * Tracks log probabilities for scoring sequences
    * Returns best sequence by final score

**Important Architecture Notes**:
* All sublayers use residual connections where output equals input plus sublayer transformation
* Layer normalization applied after each residual connection
* Dropout applied to attention weights and feedforward activations
* Encoder and decoder can have different numbers of layers
* The model uses separate embedding layers for source and target vocabularies in this implementation
* Output projection layer maps decoder output to target vocabulary logits

### Training and Evaluation (`main.py`) - TO IMPLEMENT
You need to implement the main training script with training loops, evaluation loops, and BLEU score computation. This file orchestrates all components together.

#### Required Functions:

**1. `train_epoch(model, dataloader, optimizer, loss_fn, device)`**:

Implements one epoch of training with teacher forcing.

**Steps**:
1. Set model to training mode: `model.train()`
2. Move model to device (GPU/CPU)
3. Initialize `total_loss = 0.0`
4. Iterate through batches from dataloader:
   * Each batch contains: `(encoder_input_ids, decoder_input_ids, label_ids)`
   * Move all tensors to device
   * **Forward pass**: `logits = model(encoder_input_ids, decoder_input_ids)`
     - Output shape: `(batch_size, tgt_seq_len, vocab_size)`
   * **Reshape for loss computation**:
     - Flatten logits: `(batch_size * tgt_seq_len, vocab_size)`
     - Flatten labels: `(batch_size * tgt_seq_len,)`
     - This is required by PyTorch's `CrossEntropyLoss`
   * **Compute loss**: `loss = loss_fn(flattened_logits, flattened_labels)`
     - Loss function automatically ignores padding tokens if `ignore_index=pad_id`
   * **Backward pass**:
     - `optimizer.zero_grad()` - Clear previous gradients
     - `loss.backward()` - Compute gradients
     - `optimizer.step()` - Update parameters
   * Accumulate loss: `total_loss += loss.item()`
5. Return average loss: `total_loss / len(dataloader)`

**Tip**: Use `tqdm(dataloader, desc='Training')` for a progress bar

**2. `test_epoch(model, dataloader, loss_fn, device)` or `eval_epoch(...)`**:

Evaluates model on validation or test set (without updating parameters).

**Steps**:
1. Set model to evaluation mode using the eval method
2. Move model to device
3. Initialize total_loss to zero
4. Use PyTorch's no_grad context manager to disable gradient computation and save memory
5. Iterate through batches (same as training):
   * Forward pass and loss computation (identical to training)
   * **No backward pass or optimizer step**
   * Accumulate loss
6. Return average loss by dividing total_loss by the number of batches

**3. BLEU Score Computation**:

Evaluate generation quality on test set using BLEU metric.

**Steps**:
1. Import the sentence_bleu and SmoothingFunction from nltk.translate.bleu_score module
2. Set model to eval mode and move to device
3. Initialize counters for total_bleu and num_samples to zero
4. Iterate through test batches:
   * **Generate sequences**: Call the model's generate method with source IDs, BOS ID, EOS ID, maximum length, and chosen strategy (greedy or beam_search)
   * For each sample in batch:
     - **Decode predictions**: Convert generated token IDs to word tokens using the tokenizer's decode method
     - **Remove special tokens**: Filter out BOS, EOS, and PAD tokens from predictions
     - **Decode ground truth**: Convert label IDs to tokens
     - **Remove padding** from ground truth labels
     - **Compute BLEU**: Use sentence_bleu with the reference as a list containing one sequence, the hypothesis as the prediction, weights set to favor unigrams (1.0 for unigrams, 0 for others), and apply a smoothing function for better scores on short sequences
     - Accumulate the score
5. Return average BLEU by dividing total_bleu by num_samples

**4. `main()` Function**:

Main entry point that ties everything together. Use the `click` library for command-line arguments.

**Function Signature**:
Decorate the main function with click.command() and define three click.argument decorators for train_file, dev_file, and test_file. Each argument should use click.Path with exists=True to validate file paths.

**Steps**:
1. **Define hyperparameters**: Set values for epochs, learning rate, batch size, max_src_len, max_tgt_len, and model hyperparameters including d_model, num_heads, d_ff, num_enc_layers, num_dec_layers, and dropout.

2. **Build tokenizer**: Create a Tokenizer instance and call its from_file method with the training file path. This builds the vocabulary from training data only.

3. **Create datasets**: Instantiate three SeqPairDataset objects for training, development, and test sets, each with their respective file paths, the tokenizer, and maximum sequence lengths.

4. **Create dataloaders**: Create PyTorch DataLoader objects for all three datasets. The training loader should shuffle data, while dev and test loaders should not shuffle.

5. **Initialize model**: Create an EncoderDecoder instance with source and target vocabulary sizes from the tokenizer, along with all architectural hyperparameters like d_model, num_heads, d_ff, layer counts, max_len, dropout, and the padding token index.

6. **Initialize optimizer and loss function**: Create an Adam optimizer with the model's parameters and specified learning rate. Create a CrossEntropyLoss function with ignore_index set to the padding token ID.

7. **Training loop**: Iterate over the specified number of epochs. For each epoch, call train_epoch and test_epoch functions to get training and validation losses, then print the results.

8. **Final evaluation**: Generate predictions on the test set using the trained model, compute and report the BLEU score, and optionally print example predictions alongside ground truth for qualitative assessment.

**Entry Point**: Use the standard Python idiom to call the main function when the script is executed directly.

#### Hyperparameters to Experiment With:
* **Model depth**: `num_enc_layers`, `num_dec_layers` (1, 2, 4)
* **Model dimension**: `d_model` (128, 256)
* **Attention heads**: `num_heads` (2, 4, 8) - must divide `d_model`
* **Feedforward dimension**: `d_ff` (256, 512)
* **Other parameters**: Choose one combination of (`batch_size`, `dropout`, `max_len`) to experiment with above hyperparameters

#### Tips:
* Start with small hyperparameters to verify your implementation works
* Monitor both training and validation loss to detect overfitting
* Save model checkpoints: `torch.save(model.state_dict(), 'checkpoint.pt')`
* Use GPU if available: Check `torch.cuda.is_available()`
* Print example outputs periodically to qualitatively assess progress

**Testing Your Implementation**:
1. Start with a tiny dataset (10-20 samples) to verify data loading
2. Run 1-2 epochs with small model (`d_model=64`, `num_layers=1`) to check training works
3. Verify loss decreases over epochs
4. Check that generated outputs are reasonable (even if not perfect)
5. Once working, scale up to full dataset and larger models

### Experiments
Design and run experiments to understand how different model components and configurations affect performance. You are required to complete all three experiments below.

#### Experiment 1: Positional Encoding Strategies
Compare sinusoidal positional encoding (already provided) with learnable positional embeddings.

**Background**:
The provided `PosEncoding` class uses fixed sinusoidal functions to encode position information. An alternative approach is to use learnable positional embeddings as trainable parameters, similar to word embeddings.

**Tasks**:

1. **Baseline**: Use the provided sinusoidal positional encoding. The `PosEncoding` class is already implemented in `model.py` and used in both `Encoder` and `Decoder` classes. No modification needed for this baseline.

2. **Alternative**: Modify `model.py` to use learnable positional embeddings. Create a new class `LearnablePosEncoding` that uses PyTorch's Embedding layer to learn position representations. The forward method should generate position indices and add the learned position embeddings to the input. Replace the `PosEncoding` instantiation with `LearnablePosEncoding` in both the `Encoder` and `Decoder` classes.

3. **Training Protocol**: Train both models with identical hyperparameters including the same learning rate, batch size, number of epochs, model architecture, and random seed for reproducibility. Save training and validation loss curves for both models and evaluate both on the test set using BLEU score.

**What to Report**:
* Training and validation loss curves plotted on the same graph for easy comparison
* Test set BLEU scores for both approaches
* Number of trainable parameters for each model
* Example outputs showing qualitative differences between the two methods


#### Experiment 2: Decoding Algorithms
Compare greedy decoding and beam search for text generation.

**Background**:
During inference, the model generates sequences autoregressively (one token at a time). The `generate()` method in `EncoderDecoder` supports two decoding strategies:
* **Greedy decoding**: Always selects the highest probability token at each step
* **Beam search**: Maintains multiple candidate sequences and selects the best overall sequence by score

**Tasks**:

1. **Train a single model** to convergence using your best hyperparameters from initial experiments. Save the trained model checkpoint.

2. **Generate predictions using greedy decoding**: Set the strategy parameter to "greedy" when calling the model's generate method.

3. **Generate predictions using beam search** with three different beam widths: 3, 5, and 10. Set the strategy parameter to "beam_search" and specify the beam_width parameter for each experiment.

4. **Collect metrics** for each decoding strategy:
   * BLEU score on the test set
   * Generation time per sample (use Python's time module)
   * Average sequence length

**What to Report**:
* Comparison of BLEU scores across all four strategies (greedy plus three beam search configurations)
* Time efficiency comparison showing how much slower beam search is relative to greedy decoding
* Five to ten example outputs for the same inputs showing source sequence, ground truth target, greedy output, and beam search outputs with different beam widths


#### Experiment 3: Model Architecture Variants
Explore how different architectural choices affect model performance.

You must experiment with **at least two** of the following dimensions:

**a) Number of Attention Heads**:
* Train models with different numbers of heads: 2, 4, 8
* Keep `d_model` fixed (ensure `d_model % num_heads == 0`)
* Analyze: How does multi-head attention affect learning and generation quality?

**b) Encoder/Decoder Depth**:
* Train models with different numbers of layers: 1, 2, 4
* Keep encoder and decoder depths the same
* Analyze: Does deeper architecture always improve performance? What are the trade-offs?

**c) Residual Connections**:
* Modify the model to remove residual connections in encoder/decoder layers
* Compare with the baseline that includes residual connections
* Analyze: How critical are residual connections for training deep Transformers?

**d) Attention Mechanism Variants**:
* Implement an alternative to scaled dot-product attention, such as:
  * Additive attention (Bahdanau attention)
  * Multiplicative attention without scaling
  * Attention with different similarity functions
* Compare with the provided scaled dot-product attention
* Analyze: How does the attention mechanism choice affect performance?

**Note**: Options (c) and (d) require modifying `model.py`. If you choose these, clearly document your changes.

**Analysis**:
* Which architectural configuration works best for this task?
* How do different choices affect training time and memory usage?
* What patterns do you observe in the relationship between model capacity and performance?
* Are there diminishing returns or overfitting with larger models?

---

**General Requirements for All Experiments**:
* Keep other hyperparameters constant when comparing configurations
* Report training and validation loss curves for each configuration
* Report test set BLEU scores
* Include example outputs demonstrating qualitative differences
* Provide quantitative results in tables
* Provide insightful analysis explaining your observations
* Discuss any unexpected results or failure cases

### Write-up
The writeup needs to follow an academic writing convention including below components:
* **Introduction**: Describe the task, motivation, and your approach
* **Methods**:
  * Explain the Transformer architecture
  * Describe attention mechanism and masking
  * Detail your implementation choices
  * Specify hyperparameters and training procedures
* **Results**:
  * Present experimental results with tables and figures
  * Include loss curves and BLEU scores
  * Show example outputs (input, target, prediction)
* **Analysis**:
  * Interpret your results
  * Discuss what worked well and what didn't
  * Analyze failure cases
  * Compare different configurations
* **Conclusion**:
  * Summarize findings
  * Discuss limitations
  * Suggest future improvements


## Submission
When you're ready to submit, push your branch to GitHub and submit a pull request (PR). The submission
includes two things:
1. Implemented codes (all `.py` files)
2. A writeup as a `.pdf` file

> [!IMPORTANT]
> You'll not receive any extension on this project, your quota of late days will be accordingly reduced if you submit late.

## Rubrics
The grading is based on below parts:
* Complete and successful implementations (30)
  * Dataset (15)
  * Training & evaluation (15)
* Experiments (60)
  * Experiment 1: Positional encoding strategies (15)
  * Experiment 2: Decoding algorithms (20)
  * Experiment 3: Model architecture variants (20)
  * Quality of analysis and insights across all experiments (5)
* Writeup (10)
