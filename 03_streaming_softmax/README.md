# Streaming softmax

Deeply understanding the softmax function is important for AI computing. Its properties
allow for closely-related vector functions - functions which map vectors to scalars -
such as:

* Sum of scaled exponentials (the denominator of softmax)
* Softmax dot

to be computed "online": that is, just one batch of elements at a time. This is even true
when we first need to scale softmax by subtracting the maximum value, which is done to
ensure numerical stability.

- `01_sum_of_exponentials_simple_example.py` – A simple, introductory example that
  demonstrates the streaming sum of exponentials concept with a small vector and
  verbose output.
- `02_sum_of_exponentials_large_example.py` – A larger-scale example that applies
  the same streaming logic to a vector of 1000 random integers.
- `03_softmax_dot_product_streaming_example.py` – An example that computes a
  "softmax dot product" in a streaming way, using `torch` tensors.

## Example outputs

### 01\_sum\_of\_exponentials\_simple\_example

```bash
python 03_streaming_softmax/01_sum_of_exponentials_simple_example.py   
All values: [0.5, 0.6, 0.0, 0.2, 0.8, 0.1]

Streaming computation (block_size=3):

First block max m_old = 0.6
exp(x - m_old): [0.9048, 1.0000, 0.5488]
Running sum after first block (l_old) = 2.453649

Second block max m_block = 0.8
exp(x - m_block): [0.5488, 1.0000, 0.4966]
Block sum (before rescaling) = 2.045397

=== Rescaling step ===
New global max m_new = 0.8
scale_accumulation = exp(m_old - m_new) = 0.818731

Rescaled prior sum   = 2.008878
New block sum        = 2.045397
Running sum (online) = 4.054275

Streaming sum of scaled exponentials: 4.054275
Full sum of scaled exponentials: 4.054275

Difference: 0.00000000
✅ SUCCESS: Streaming result matches full sum.
```

### 02\_sum\_of\_exponentials\_large\_example

```bash
python 03_streaming_softmax/02_sum_of_exponentials_large_example.py    
Vector length                        = 1,000
Block size                           = 50
Full sum of scaled exponentials      = 18.539417
Streaming sum of scaled exponentials = 18.539417
Difference                           = 0.00000000
✅ SUCCESS: Streaming sum of scaled exponentials matches full sum.
```

### 03\_softmax\_dot\_product\_streaming\_example

```bash
python 03_streaming_softmax/03_softmax_dot_product_streaming_example.py
Vector length                        = 100
Block size                           = 10
Full softmax · V: 0.959081
Streaming softmax · V: 0.959081
Difference                           = 0.00000008
✅ SUCCESS: Streaming result matches full softmax-dot.
```

### 03\_flash\_attention\_streaming\_softmax

```bash
python 03_flash_attention_streaming_softmax.py

Batch shape: torch.Size([2, 8, 3])

--- Standard MHA Output ---
tensor([[ 0.3969, -0.0506, -0.2503,  0.4973, -0.1616,  0.1172],
        [ 0.3886, -0.0391, -0.2593,  0.5047, -0.1559,  0.1111]],
       grad_fn=<SliceBackward0>)

--- Flash MHA Output ---
tensor([[ 0.3969, -0.0506, -0.2503,  0.4973, -0.1616,  0.1172],
        [ 0.3886, -0.0391, -0.2593,  0.5047, -0.1559,  0.1111]],
       grad_fn=<SliceBackward0>)

Max Difference: 0.00000006
✅ SUCCESS: The Flash algorithm matches the Standard algorithm!
```


