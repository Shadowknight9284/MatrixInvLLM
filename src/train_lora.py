import os
import json
import asyncio
import logging
import time
import numpy as np
import yaml
import torch
from pathlib import Path

# Tinker SDK imports
try:
    import tinker
    from tinker import ServiceClient, AdamParams, ModelInput, EncodedTextChunk, TensorData
    from tinker_cookbook.supervised.data import datum_from_model_input_weights
except ImportError:
    tinker = None
    ServiceClient = None
    AdamParams = None
    ModelInput = None
    EncodedTextChunk = None
    TensorData = None
    datum_from_model_input_weights = None


logger = logging.getLogger(__name__)

async def train_lora():
    """
    Supervised LoRA fine-tuning for matrix inversion using the Tinker API.
    """
    # 1. Load Configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Setup paths
    outputs_dir = Path(config["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 2. Initialize Tinker client
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        logger.error("TINKER_API_KEY not found in environment.")
        return

    if ServiceClient is None:
        logger.error("tinker-python SDK not installed. Please run 'pip install tinker'.")
        return
        
    client = ServiceClient(api_key=api_key)
    
    # 3. Create LoRA training client
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    base_model = model_cfg.get("base_model", "meta-llama/Llama-3.2-3B")
    
    logger.info(f"Connecting to Tinker and initializing training client for {base_model}...")
    try:
        # Note: Using create_lora_training_client_async as per internal documentation
        training_client = await client.create_lora_training_client_async(
            base_model=base_model,
            rank=lora_cfg.get("rank", 16)
        )
    except Exception as e:
        logger.error(f"Failed to initialize training client: {e}")
        return
    
    # 4. Load training dataset
    train_file = config["paths"]["train_file"]
    if not os.path.exists(train_file):
        logger.error(f"Training data not found at {train_file}. Run generate_matrices.py first.")
        return
        
    logger.info(f"Loading training examples from {train_file}...")
    sft_examples = []
    with open(train_file, "r") as f:
        for line in f:
            item = json.loads(line)
            sft_examples.append({"input_text": item["input_text"], "target_text": item["target_text"]})

    # 5. Tokenize using the training client's tokenizer
    tokenizer = training_client.get_tokenizer()
    max_seq_length = config["training"].get("max_seq_length", 1024)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        logger.warning("Tokenizer has no eos_token_id — model will not learn a stop signal.")

    def encode_sft_example(ex):
        input_ids = tokenizer.encode(ex["input_text"])
        # Suppress BOS on the target: the model must NOT learn to re-emit
        # <|begin_of_text|> mid-sequence (the root cause of repeated-block loops).
        try:
            target_ids = tokenizer.encode(ex["target_text"], add_special_tokens=False)
        except TypeError:
            # Fallback for tokenizer wrappers that don't accept add_special_tokens.
            target_ids = tokenizer.encode(ex["target_text"])
            if target_ids and target_ids[0] == tokenizer.bos_token_id:
                target_ids = target_ids[1:]

        # Append EOS so the model learns exactly where to stop.
        if eos_token_id is not None:
            target_ids = target_ids + [eos_token_id]

        all_tokens = input_ids + target_ids
        # Weight 0 on the input (conditioning), weight 1 on the target (what to learn)
        weights = torch.tensor([0.0] * len(input_ids) + [1.0] * len(target_ids))
        model_input = ModelInput(chunks=[EncodedTextChunk(tokens=all_tokens)])
        return datum_from_model_input_weights(model_input, weights, max_length=max_seq_length)

    logger.info(f"Tokenizing and preparing {len(sft_examples)} examples...")
    training_data = [encode_sft_example(ex) for ex in sft_examples]
    
    # 6. Run Training Loop
    train_cfg = config.get("training", {})
    opt_cfg = config.get("optimizer", {})
    
    max_steps = train_cfg.get("max_steps", 400)
    batch_size = train_cfg.get("batch_size", 16)
    lr = float(opt_cfg.get("lr", 1.0e-4))
    
    logger.info(f"Starting LoRA training run: steps={max_steps}, batch_size={batch_size}, lr={lr}")
    
    current_time = time.time()
    for step in range(max_steps):
        t0 = time.time()
        
        # Sample mini-batch with replacement
        indices = np.random.choice(len(training_data), batch_size, replace=True)
        batch = [training_data[i] for i in indices]
        
        try:
            # 6a. Forward-Backward Pass (calculates gradients)
            # Note: Using async methods as per Tinker tutorial patterns
            fwdbwd_future = await training_client.forward_backward_async(batch, "cross_entropy")
            
            # 6b. Optimizer Step (applies gradients)
            optim_future = await training_client.optim_step_async(
                AdamParams(
                    learning_rate=lr,
                    weight_decay=opt_cfg.get("weight_decay", 0.01)
                )
            )
            
            # 6c. Wait for results (server pipelining ensures efficiency)
            fwdbwd_result = await fwdbwd_future.result_async()
            optim_result = await optim_future.result_async()
            
            step_elapsed = time.time() - t0
            
            # 6d. Periodic Logging
            if step % train_cfg.get("logging_steps", 10) == 0:
                # Approximate loss computation from result metadata if available
                # (Actual loss extraction varies by Tinker version, using a generic log profile)
                logger.info(f"Step {step:3d}/{max_steps} | Time: {step_elapsed:.2f}s | Status: OK")
                
        except Exception as e:
            logger.error(f"Training interrupted at step {step}: {e}")
            break

    # 7. Persist fine-tuned weights and model ID
    logger.info("Training complete. Exporting weights to Tinker ecosystem...")
    try:
        # LoRA weights are saved under a unique name for subsequent inference
        export_name = f"matrix-inv-lora-{int(time.time())}"
        
        # Use persistent weights save instead of ephemeral
        # save_weights_for_sampler_async returns an APIFuture, which itself needs to be awaited.
        save_res_future = await training_client.save_weights_for_sampler_async(name=export_name)
        save_res = await save_res_future
        final_model_path = save_res.path
        
        model_id_path = outputs_dir / "finetuned_model_id.txt"
        with open(model_id_path, "w") as f:
            f.write(final_model_path)
            
        logger.info(f"SUCCESS: Fine-tuned model path persisted to {model_id_path}: {final_model_path}")
        
    except Exception as e:
        logger.error(f"Failed to persist fine-tuned weights: {e}")

if __name__ == "__main__":
    # Ensure TINKER_API_KEY is available
    if not os.environ.get("TINKER_API_KEY"):
        # Attempt to load from .env if available
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
            
    asyncio.run(train_lora())
