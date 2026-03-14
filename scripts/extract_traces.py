"""Step 1: Extract real h_pre_attn traces from DeepSeek-V2-Lite.

Runs the model on long prompts and saves per-layer hidden states
and true gate routing decisions. These are used to:
  - Train real linear probes (Step 2)
  - Drive the closed-loop benchmark with real data (Step 3)

Memory-efficient: processes layers in batches to avoid OOM when
hooking all 26 MoE layers simultaneously.

Usage:
    python scripts/extract_traces.py [--num-prompts 10] [--max-len 2048]
"""
import gc
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path


def get_long_prompts(num_prompts: int, target_tokens: int) -> list[str]:
    """Generate long English prompts by repeating diverse text."""
    base_texts = [
        "The development of large language models has transformed natural language processing. "
        "These models use transformer architectures with attention mechanisms that allow them to "
        "capture long-range dependencies in text. The scaling laws suggest that larger models "
        "trained on more data consistently improve performance across benchmarks. ",

        "In distributed computing systems, the challenge of minimizing communication overhead "
        "while maximizing computational throughput remains central. Expert parallelism divides "
        "model parameters across devices, requiring all-to-all communication patterns that can "
        "become bottlenecks at scale. Network topology and bandwidth constraints determine the "
        "practical limits of scaling. ",

        "Mixture of Experts models achieve computational efficiency by activating only a subset "
        "of parameters for each input token. The routing mechanism, typically a learned gate "
        "network, determines which experts process each token. Load balancing across experts "
        "is critical for training stability and inference efficiency. The sparse activation "
        "pattern creates unique challenges for distributed deployment. ",

        "Modern GPU architectures provide massive parallel computation through thousands of "
        "streaming multiprocessors. The memory hierarchy includes registers, shared memory, "
        "L2 cache, and high-bandwidth memory. Optimizing kernel launches and minimizing "
        "synchronization points are essential for achieving peak hardware utilization. ",

        "The attention mechanism computes a weighted sum over value vectors, where weights "
        "are determined by the compatibility between query and key vectors. Flash attention "
        "reduces memory footprint by computing attention in blocks, avoiding materialization "
        "of the full attention matrix. This enables processing of much longer sequences. ",

        "Reinforcement learning from human feedback has become a standard technique for aligning "
        "language models with human preferences. The reward model learns to predict human ratings "
        "of model outputs, which guides policy optimization through proximal policy algorithms. "
        "Constitutional AI extends this by training models to critique their own responses. ",

        "Quantization techniques reduce model memory footprint by representing weights and "
        "activations with fewer bits. Post-training quantization applies compression without "
        "retraining, while quantization-aware training incorporates precision constraints during "
        "optimization. INT8 and FP8 formats balance accuracy with computational efficiency. ",

        "The transformer architecture processes sequences through self-attention layers that "
        "compute pairwise token interactions. Multi-head attention allows the model to attend "
        "to information from different representation subspaces at different positions. Layer "
        "normalization and residual connections stabilize training of deep networks. ",

        "Distributed training of large models requires sophisticated parallelism strategies "
        "including data parallelism, tensor parallelism, pipeline parallelism, and expert "
        "parallelism. ZeRO optimization partitions optimizer states, gradients, and parameters "
        "across data parallel ranks to reduce memory redundancy. Communication efficiency is "
        "critical for achieving near-linear scaling. ",

        "Speculative decoding accelerates autoregressive generation by using a smaller draft "
        "model to propose multiple tokens that are then verified in parallel by the target "
        "model. The acceptance rate depends on the alignment between draft and target model "
        "distributions. This technique provides lossless speedup while maintaining output "
        "quality identical to standard autoregressive decoding. ",
    ]

    prompts = []
    words_per_token = 1.3
    target_words = int(target_tokens * words_per_token)

    for i in range(num_prompts):
        base = base_texts[i % len(base_texts)]
        words_in_base = len(base.split())
        repeats = max(1, target_words // words_in_base + 1)
        prompt = (base * repeats)[:target_words * 6]
        prompts.append(prompt)

    return prompts


def extract_traces(args):
    """Extract hidden states and gate outputs from the model.

    Memory-efficient strategy:
      - Process layers in batches of --layers-per-batch (default 6)
      - Each batch hooks only a few layers, runs all prompts, saves to disk
      - This keeps GPU memory usage bounded regardless of total MoE layers
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    # Identify MoE layer indices and gate weights
    moe_layers = []
    gate_weights = {}
    for name, param in model.named_parameters():
        if name.endswith(".mlp.gate.weight"):
            parts = name.split(".")
            layer_idx = int(parts[2])
            num_experts = param.shape[0]
            if num_experts > 512:
                continue
            moe_layers.append(layer_idx)
            gate_weights[layer_idx] = param.detach().cpu()
            print(f"  Found MoE gate: {name}, shape={param.shape}")

    moe_layers = sorted(set(moe_layers))
    print(f"Found {len(moe_layers)} MoE layers: {moe_layers[:5]}...{moe_layers[-3:]}")

    # Use ALL MoE layers as anchor layers (probe every layer for max overlap)
    anchor_layers = moe_layers
    print(f"Anchor layers for probing: all {len(anchor_layers)} MoE layers")

    for layer_idx in anchor_layers:
        torch.save(gate_weights[layer_idx], save_dir / f"gate_weight_layer{layer_idx}.pt")

    # Tokenize prompts once (reused across batches)
    prompts = get_long_prompts(args.num_prompts, args.max_len)
    tokenized_prompts = []
    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=args.max_len,
            truncation=True,
        )
        tokenized_prompts.append(inputs)

    print(f"Prepared {len(tokenized_prompts)} prompts (target ~{args.max_len} tokens each)")

    # Process layers in batches to limit memory usage
    batch_size = args.layers_per_batch
    layer_batches = [
        anchor_layers[i:i + batch_size]
        for i in range(0, len(anchor_layers), batch_size)
    ]
    print(f"Processing {len(anchor_layers)} layers in {len(layer_batches)} batches "
          f"(batch_size={batch_size})")

    for batch_idx, batch_layers in enumerate(layer_batches):
        print(f"\n── Batch {batch_idx + 1}/{len(layer_batches)}: layers {batch_layers} ──")

        # Accumulate traces for this batch of layers
        batch_traces = {
            layer_idx: {"h_pre": [], "true_gate_ids": []}
            for layer_idx in batch_layers
        }

        # Register hooks for this batch only
        captured = {}
        hooks = []

        def make_pre_attn_hook(layer_idx):
            def hook_fn(module, args, kwargs=None):
                if isinstance(args, tuple) and len(args) > 0:
                    h = args[0]
                else:
                    return
                captured.setdefault(layer_idx, []).append(
                    h.detach().cpu().to(torch.float16)
                )
            return hook_fn

        def make_post_attn_hook(layer_idx):
            def hook_fn(module, args, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                captured.setdefault(f"{layer_idx}_post", []).append(
                    h.detach().cpu().to(torch.float16)
                )
            return hook_fn

        for layer_idx in batch_layers:
            layer = model.model.layers[layer_idx]
            hooks.append(layer.input_layernorm.register_forward_hook(
                make_pre_attn_hook(layer_idx)
            ))
            hooks.append(layer.self_attn.register_forward_hook(
                make_post_attn_hook(layer_idx)
            ))

        # Run all prompts with this batch's hooks
        for pi, inputs in enumerate(tokenized_prompts):
            captured.clear()

            inputs_gpu = {k: v.to(model.device) for k, v in inputs.items()}
            actual_len = inputs_gpu["input_ids"].shape[1]

            with torch.no_grad():
                _ = model(**inputs_gpu, use_cache=False)

            for layer_idx in batch_layers:
                if layer_idx in captured:
                    h_pre = captured[layer_idx][0].squeeze(0)

                    post_key = f"{layer_idx}_post"
                    h_post = captured[post_key][0].squeeze(0) if post_key in captured else h_pre

                    gw = gate_weights[layer_idx].to(h_post.dtype)
                    gate_logits = F.linear(h_post.float(), gw.float())
                    true_topk = torch.topk(
                        gate_logits, k=min(6, gate_logits.shape[-1]), dim=-1
                    )

                    batch_traces[layer_idx]["h_pre"].append(h_pre)
                    batch_traces[layer_idx]["true_gate_ids"].append(true_topk.indices.cpu())

            if (pi + 1) % 5 == 0 or pi == len(tokenized_prompts) - 1:
                print(f"    Prompt {pi + 1}/{len(tokenized_prompts)}: {actual_len} tokens")

        # Remove hooks
        for h in hooks:
            h.remove()

        # Save this batch's traces to disk immediately
        for layer_idx in batch_layers:
            data = batch_traces[layer_idx]
            if not data["h_pre"]:
                print(f"  WARNING: No data captured for layer {layer_idx}")
                continue

            h_pre_all = torch.cat(data["h_pre"], dim=0)
            true_ids_all = torch.cat(data["true_gate_ids"], dim=0)

            trace_path = save_dir / f"traces_layer{layer_idx}.pt"
            torch.save({
                "h_pre": h_pre_all,
                "true_gate_ids": true_ids_all,
                "num_tokens": h_pre_all.shape[0],
                "hidden_dim": h_pre_all.shape[1],
                "num_experts": gate_weights[layer_idx].shape[0],
            }, trace_path)

            print(f"  Layer {layer_idx}: saved {h_pre_all.shape[0]} tokens, "
                  f"hidden={h_pre_all.shape[1]}, experts={gate_weights[layer_idx].shape[0]}")

        # Free batch memory
        del batch_traces, captured
        gc.collect()
        torch.cuda.empty_cache()

    # Save metadata
    meta = {
        "model": model_name,
        "anchor_layers": anchor_layers,
        "all_moe_layers": moe_layers,
        "num_prompts": args.num_prompts,
        "max_len": args.max_len,
        "hidden_dim": config.hidden_size,
        "num_experts": len(moe_layers) > 0 and gate_weights[moe_layers[0]].shape[0] or 0,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTraces saved to {save_dir}/")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    parser.add_argument("--num-prompts", type=int, default=20,
                        help="Number of prompts for diverse training data")
    parser.add_argument("--max-len", type=int, default=2048,
                        help="Max token length per prompt (reduced from 4096 for memory)")
    parser.add_argument("--save-dir", default="probes/traces")
    parser.add_argument("--layers-per-batch", type=int, default=6,
                        help="Number of layers to hook simultaneously (lower = less memory)")
    args = parser.parse_args()

    extract_traces(args)
