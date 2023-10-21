# DEV LOG

- fsdp, after separating out the `setup` calls for model and optim, throws a known-ish error about uniform requires-grads, where the offending seems to be that positional embed in the dummy trasnformer
- tested out some bsz configs to try and OOM, able to, but might be some odd autoscaling happening w/ precision vs bsz? idk
- manually setting up Fabric parameters
- have to launch an srun job with name as "interactive"
- had to fix the minimal transformer example to get the starting code to run
- warning a matmul precision probably only for demo
    - You are using a CUDA device ('NVIDIA RTX A5000') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

## Transformers

This example contains a simple training loop for next-word prediction with a [Transformer model](https://arxiv.org/abs/1706.03762) on a subset of the [WikiText2](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) dataset.
It is a simplified version of the [official PyTorch example](https://github.com/pytorch/examples/tree/main/word_language_model).

### Train with Fabric

```bash
# CPU
lightning run model --accelerator=cpu train.py

# GPU (CUDA or M1 Mac)
lightning run model --accelerator=gpu train.py

# Multiple GPUs
lightning run model --accelerator=gpu --devices=4 train.py
```
