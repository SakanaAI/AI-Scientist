import pandas as pd
from sklearn.model_selection import KFold
import torch
from transformer_lens import HookedTransformer
from typing import Callable
from tqdm import tqdm
import os
import argparse
import datasets

def generate_training_prompts(
    data: pd.DataFrame,
    n_pairs_per_training_prompt: int = 10,
    separator: str = ', '
) -> list[str]:
    r"""
    Generates training prompts from the data.

    Args:
        data (pd.DataFrame): The data to generate training prompts from.
            Must have 'input' and 'target' columns.
        n_pairs_per_training_prompt (int): The number of pairs per training prompt.
        separator (str): The separator to use between pairs.
    Returns:
        list[str]: A list of training prompts.
            Each training prompt is a string of the form:
            "input1:target1{separator}input2:target2{separator}...\ninputN:"
            In particular, the last target is not included, and it ends on a colon.
    """
    n_training_prompts = len(data) // n_pairs_per_training_prompt

    if n_training_prompts == 0:
        raise ValueError("Not enough data to generate training prompts.")

    training_prompts: list[str] = []
    for i in range(n_training_prompts):
        training_prompt = ""
        for j in range(n_pairs_per_training_prompt-1):
            training_prompt += f"{data.iloc[i*n_pairs_per_training_prompt + j]['input']}"
            training_prompt += ':'
            training_prompt += f"{data.iloc[i*n_pairs_per_training_prompt + j]['target']}"
            training_prompt += separator
        training_prompt += f"{data.iloc[i*n_pairs_per_training_prompt + n_pairs_per_training_prompt-1]['input']}"
        training_prompt += ':'
        training_prompts.append(training_prompt)
    
    return training_prompts

def generate_test_prompts(
    data: pd.DataFrame,
) -> list[str]:
    r"""
    Generates test prompts from the data.

    Args:
        data (pd.DataFrame): The data to generate training prompts from.
            Must have 'input'

    Returns:
        list[str]: A list of test prompts.
            Each training prompt is a string of the form "input:"
    """
    test_prompts: list[str] = []
    for i in range(len(data)):
        test_prompts.append(f"{data.iloc[i]['input']}:")
    
    return test_prompts

def get_cv_splits(data: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generates cross-validation splits from the data.

    Args:
        data (pd.DataFrame): The data to generate cross-validation splits from.
        n_splits (int): The number of splits to generate.
        random_state (int): The random seed to use for the splits.
    Returns:
        list[tuple[pd.DataFrame, pd.DataFrame]]: A list of tuples, each containing a training and a test set.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_splits = []
    for train_idx, test_idx in kf.split(data):
        train_df = data.iloc[train_idx].copy()
        test_df = data.iloc[test_idx].copy()
        cv_splits.append((train_df, test_df))
    
    return cv_splits

# only used for debugging
def check_torch_gpu_memory() -> None:
    """
    Checks the memory usage of the GPU at the current time. Only used for debugging.
    No args or return value, just prints information.
    """
    # Total memory available on the GPU
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # in GB
    
    # Memory reserved by PyTorch's allocator
    reserved_memory = torch.cuda.memory_reserved() / 1e9  # in GB
    
    # Memory actually allocated by PyTorch (subset of reserved)
    allocated_memory = torch.cuda.memory_allocated() / 1e9  # in GB
    
    print(f"Total GPU Memory: {total_memory:.4f} GB")
    print(f"Reserved by PyTorch: {reserved_memory:.4f} GB")
    print(f"Allocated by PyTorch: {allocated_memory:.4f} GB")

def extract_activations_last_token(
        model: HookedTransformer,
        prompts: list[str],
        extraction_layers: list[int],
        batch_size: int = 64,
) -> dict[int, torch.Tensor]:
    """
    Extract activations for the last token of each prompt from specific layers of the model.
    
    Processes prompts in batches to avoid memory issues.
    If you get an error about memory, try reducing the batch size.

    Parameters:
    model (HookedTransformer): The model used for generating text.
    prompts (list): List of prompts to extract activations for.
    extraction_layers (list): The layers from which activations are extracted.
    batch_size (int): Number of prompts to process at once.

    Returns:
    dict[int, torch.Tensor]: A dictionary where each key is a layer number and each value is the
        activations for the last token of each prompt. Shape: (n_prompts, d_model).
    """
    activations_dict = {layer: [] for layer in extraction_layers}
    names = [f"blocks.{layer}.hook_resid_pre" for layer in extraction_layers]
    
    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n in names)
        
        with model.hooks(fwd_hooks=caching_hooks):
            model.tokenizer.padding_side = "left"
            _ = model(batch_prompts)
            
        for layer in extraction_layers:
            prompt_activations = cache[f"blocks.{layer}.hook_resid_pre"].detach().cpu()
            last_token_activations = prompt_activations[:, -1, :].squeeze()
            # Handle the case where there's only one prompt in the batch
            if len(batch_prompts) == 1:
                last_token_activations = last_token_activations.unsqueeze(0)
            activations_dict[layer].append(last_token_activations)
        
        # Clear CUDA cache after each batch
        torch.cuda.empty_cache()
    
    # Concatenate the batched results
    for layer in extraction_layers:
        activations_dict[layer] = torch.cat(activations_dict[layer], dim=0)
    
    return activations_dict

def average_activations(
    activations: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """
    Computes averaged activations for all layers, averaging over all prompts.

    Args:
    activations (dict[int, torch.Tensor]): A dictionary where each key is a layer number and each value is the
        activations for the last token of each prompt. Shape: (n_prompts, d_model).

    Returns:
    dict[int, torch.Tensor]: A dictionary containing the averaged activations for each layer.
        Averaging over all prompts.
        Keys are layer_index and values are the averaged activations.
        Shape of values: (d_model,).
    """
    averaged_activations: dict[int, torch.Tensor] = {}

    for layer in activations.keys():
        # Extract the last-token activations of steering examples at the specified layer
        activation = activations[layer]
        # Compute the average activations
        avg_activation = torch.mean(activation, dim=0)
        # Store the average activations in the cache
        averaged_activations[layer] = avg_activation.detach().cpu()

    return averaged_activations

def download_and_compute_background_activations(
    model: HookedTransformer,
    extraction_layers: list[int],
    num_samples: int = 100,
    batch_size: int = 16,
    cache_file: str = "background_activations.pt"
) -> dict[int, torch.Tensor]:
    """
    Downloads a sample from a medium-sized dataset and computes average activations.
    
    Args:
        model (HookedTransformer): The model to compute activations for
        extraction_layers (list[int]): The layers to extract activations from
        num_samples (int): Number of samples to use from the dataset
        batch_size (int): Batch size for processing
        cache_file (str): File to save/load the computed activations
        
    Returns:
        dict[int, torch.Tensor]: Dictionary of average activations per layer
    """
    # Check if we've already computed and cached the background activations
    if os.path.exists(cache_file):
        print(f"Loading cached background activations from {cache_file}")
        return torch.load(cache_file)
    
    print(f"Downloading and computing background activations from {num_samples} samples...")
    
    # Load a sample from the Pile dataset
    dataset = datasets.load_dataset("the_pile", "all", split="train", streaming=True)
    
    # Take a sample of the dataset
    samples = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append(sample['text'])
    
    # Process in batches
    all_activations = {layer: [] for layer in extraction_layers}
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size}")
        
        # Get activations for each text sample
        names = [f"blocks.{layer}.hook_resid_pre" for layer in extraction_layers]
        cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n in names)
        
        with model.hooks(fwd_hooks=caching_hooks):
            model.tokenizer.padding_side = "left"
            _ = model(batch)
        
        # Extract activations from all positions (not just the last token)
        for layer in extraction_layers:
            # Get activations for all tokens and flatten across batch and sequence dimensions
            activations = cache[f"blocks.{layer}.hook_resid_pre"].detach().cpu()
            # Reshape to (batch_size * seq_len, d_model)
            batch_size, seq_len, d_model = activations.shape
            flattened = activations.reshape(-1, d_model)
            all_activations[layer].append(flattened)
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    # Concatenate and average
    background_activations = {}
    for layer in extraction_layers:
        all_layer_activations = torch.cat(all_activations[layer], dim=0)
        background_activations[layer] = torch.mean(all_layer_activations, dim=0)
    
    # Cache the results
    torch.save(background_activations, cache_file)
    print(f"Background activations computed and saved to {cache_file}")
    
    return background_activations

def generate_hook_addition(steering_vector: torch.Tensor, beta: float, background_vector: torch.Tensor = None) -> Callable:
    """
    Generates a hook function to add a steering vector to the last token.
    If background_vector is provided, it will be subtracted from the steering vector.

    Parameters:
    - steering_vector (torch.Tensor): Steering vector.
    - beta (float): Scaling factor.
    - background_vector (torch.Tensor, optional): Background vector to subtract from steering vector.

    Returns:
    - function: Hook function for adding steering vector.
    """
    # If background vector is provided, subtract it from the steering vector
    if background_vector is not None:
        effective_vector = steering_vector - background_vector
        # Normalize the effective vector to have the same norm as the original steering vector
        # This helps maintain consistent scaling across different background subtractions
        original_norm = torch.norm(steering_vector)
        effective_norm = torch.norm(effective_vector)
        if effective_norm > 0:  # Avoid division by zero
            effective_vector = effective_vector * (original_norm / effective_norm)
    else:
        effective_vector = steering_vector
        
    def last_token_steering_hook(resid_pre, hook):
        # Apply steering only to the last token of each sequence in the batch
        last_token_indices = resid_pre.shape[1] - 1
        # Use broadcasting for efficiency instead of a loop
        resid_pre[:, last_token_indices, :] += effective_vector.squeeze().to(resid_pre.device) * beta

    return last_token_steering_hook

def generate_text(
        model: HookedTransformer,
        prompts: list[str],
        hooks: list[Callable],
        max_new_tokens: int = 5,
        temperature: float = 0,
) -> list[str]:
    r"""
    Generates text from the model with the given settings.

    Args:
        model (HookedTransformer): The model used for generating text.
        prompts (list): List of prompts to generate text from.
        hooks (list): List of hooks to apply to the model.
        max_new_tokens (int): The maximum number of new tokens to generate.
        temperature (float): The temperature to use for the model.
    Returns:
        list[str]: A list of generated text.
            Note that the original prompts are removed from the results.
    """
    with model.hooks(fwd_hooks=hooks):
        results = model.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            padding_side='left',
            verbose=False,
        )
    
    # remove the original prompts from the results
    results = [result[len(prompt):] for result, prompt in zip(results, prompts)]

    return results


def calculate_accuracy(predictions: list[str], targets: pd.Series) -> float:
    """
    Calculates the accuracy of the predictions, by asking if the prediction starts with the target.

    Args:
        predictions (list[str]): A list of predictions.
        targets (pd.Series): A series of targets. Typically the 'target' column of the test set dataframe.

    Returns:
        float: The accuracy of the predictions.
    """
    correct = 0
    for prediction, target in zip(predictions, targets):
        if prediction.startswith(target):
            correct += 1
    return correct / len(predictions)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run experiments with demeaned probes")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--run", type=int, default=3, help="Run number (0=baseline, 1=demeaned, 2=optimized, 3=multi-layer)")
    parser.add_argument("--samples", type=int, default=200, help="Number of background samples to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    args = parser.parse_args()
    
    PATH_TO_DATA = "antonyms.json"
    MODEL_NAME = "EleutherAI/gpt-j-6b"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = "float16"
    N_SPLITS = 3
    N_PAIRS_PER_TRAINING_PROMPT = 10
    
    # Configure experiment based on run number
    if args.run == 0:  # Baseline
        EXTRACTION_LAYERS = [10]  # Just use the best layer
        LAYER_COMBINATIONS = [[10]]
        BETAS = [1, 2, 3]
        USE_BACKGROUND_SUBTRACTION = False
        BACKGROUND_SAMPLES = 0
        RUN_NAME = "baseline"
    elif args.run == 1:  # Demeaned Probes
        EXTRACTION_LAYERS = [10]
        LAYER_COMBINATIONS = [[10]]
        BETAS = [1, 2, 3]
        USE_BACKGROUND_SUBTRACTION = True
        BACKGROUND_SAMPLES = 100
        RUN_NAME = "demeaned"
    elif args.run == 2:  # Optimized Scaling
        EXTRACTION_LAYERS = [10]
        LAYER_COMBINATIONS = [[10]]
        BETAS = [0.5, 1, 2, 3, 5, 7, 10]
        USE_BACKGROUND_SUBTRACTION = True
        BACKGROUND_SAMPLES = args.samples
        RUN_NAME = "optimized"
    else:  # Multi-Layer (Run 3)
        EXTRACTION_LAYERS = list(range(9,13))
        LAYER_COMBINATIONS = [
            [10],       # Single best layer
            [9, 10],    # Two consecutive layers
            [10, 11],   # Two consecutive layers
            [9, 11],    # Two non-consecutive layers
            [9, 10, 11] # Three layers
        ]
        BETAS = [2, 3, 5]
        USE_BACKGROUND_SUBTRACTION = True
        BACKGROUND_SAMPLES = args.samples
        RUN_NAME = "multi_layer"
    
    BATCH_SIZE = args.batch_size
    RESULTS_DIR = f"{args.out_dir}/{RUN_NAME}"
    
    print(f"Running experiment: {RUN_NAME}")
    print(f"Background subtraction: {USE_BACKGROUND_SUBTRACTION}")
    print(f"Background samples: {BACKGROUND_SAMPLES}")
    print(f"Layer combinations: {LAYER_COMBINATIONS}")
    print(f"Beta values: {BETAS}")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # load data
    data = pd.read_json(PATH_TO_DATA)
    cv_pairs = get_cv_splits(data, n_splits=N_SPLITS)

    model = HookedTransformer.from_pretrained_no_processing(
        model_name=MODEL_NAME, device=DEVICE, dtype=DTYPE
    )
    model.eval()
    
    # Download and compute background activations if needed
    background_activations = None
    if USE_BACKGROUND_SUBTRACTION:
        try:
            # Try to load from cache first
            cache_file = f"{RESULTS_DIR}/background_activations_{BACKGROUND_SAMPLES}.pt"
            if os.path.exists(cache_file):
                print(f"Loading cached background activations from {cache_file}")
                background_activations = torch.load(cache_file)
            else:
                print(f"Computing background activations with {BACKGROUND_SAMPLES} samples...")
                background_activations = download_and_compute_background_activations(
                    model, 
                    EXTRACTION_LAYERS,
                    num_samples=BACKGROUND_SAMPLES,
                    batch_size=BATCH_SIZE,
                    cache_file=cache_file
                )
                
            # Verify we have activations for all required layers
            missing_layers = [layer for layer in EXTRACTION_LAYERS if layer not in background_activations]
            if missing_layers:
                print(f"Warning: Missing background activations for layers {missing_layers}")
                # Compute missing layers if needed
                if len(missing_layers) > 0:
                    print(f"Computing missing background activations for layers {missing_layers}")
                    missing_activations = download_and_compute_background_activations(
                        model, 
                        missing_layers,
                        num_samples=BACKGROUND_SAMPLES,
                        batch_size=BATCH_SIZE,
                        cache_file=f"{RESULTS_DIR}/background_activations_missing.pt"
                    )
                    # Merge with existing activations
                    for layer in missing_layers:
                        background_activations[layer] = missing_activations[layer]
                    # Save updated cache
                    torch.save(background_activations, cache_file)
                    
        except Exception as e:
            print(f"Error computing background activations: {e}")
            print("Continuing without background subtraction")
            background_activations = None
            USE_BACKGROUND_SUBTRACTION = False

    # create a dataframe to store the results
    results_df = pd.DataFrame(columns=['split', 'layers', 'beta', 'accuracy'])

    # Create a progress bar for the total number of iterations
    total_iterations = N_SPLITS * len(LAYER_COMBINATIONS) * len(BETAS)
    pbar = tqdm(total=total_iterations, desc="Overall Progress")

    for split_number in range(N_SPLITS):
        # create data
        data_train, data_test = cv_pairs[split_number]
        training_prompts = generate_training_prompts(
            data_train, n_pairs_per_training_prompt=N_PAIRS_PER_TRAINING_PROMPT, separator=', ')
        test_prompts = generate_test_prompts(data_test)

        # train steering vector on training prompts
        activations_last_token = extract_activations_last_token(model, training_prompts, EXTRACTION_LAYERS, batch_size=BATCH_SIZE)
        activations_last_token_averaged = average_activations(activations_last_token)
        for layer in EXTRACTION_LAYERS:
            assert activations_last_token[layer].shape == (len(training_prompts), model.cfg.d_model)
            assert activations_last_token_averaged[layer].shape == (model.cfg.d_model,)
        
        # test steering vector on test prompts with layer combinations
        for layer_combo in LAYER_COMBINATIONS:
            for beta in BETAS:
                # Create hooks for all layers in the combination
                hooks = []
                for layer in layer_combo:
                    # Use background-subtracted steering vector if available
                    if background_activations is not None:
                        addition_hook = generate_hook_addition(
                            steering_vector=activations_last_token_averaged[layer], 
                            beta=beta,
                            background_vector=background_activations[layer]
                        )
                    else:
                        addition_hook = generate_hook_addition(
                            steering_vector=activations_last_token_averaged[layer], 
                            beta=beta
                        )
                    hooks.append((f"blocks.{layer}.hook_resid_pre", addition_hook))
                
                # Generate text with the combined hooks
                predictions = generate_text(model, test_prompts, hooks)
                accuracy = calculate_accuracy(predictions, data_test['target'])
                
                # Convert layer combination to string for display and storage
                layers_str = '-'.join(map(str, layer_combo))
                print(f"Split {split_number}, Beta {beta}, Layers {layers_str}, Accuracy {accuracy}")
                
                # Store results
                new_row = pd.DataFrame({
                    'split': [split_number], 
                    'layers': [layers_str], 
                    'beta': [beta], 
                    'accuracy': [accuracy]
                })
                
                if len(results_df) == 0:
                    results_df = new_row
                else:
                    results_df = pd.concat([results_df, new_row], ignore_index=True)
                
                # save results
                results_df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)

                predictions_df = pd.DataFrame({
                    'prompt': test_prompts,
                    'prediction': predictions,
                    'target': data_test['target']
                })
                predictions_df.to_csv(os.path.join(RESULTS_DIR, f"predictions_{split_number}_{layers_str}_{beta}.csv"), index=False)

                pbar.update(1)

    pbar.close()
    print("All experiments completed!")

