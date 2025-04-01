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

def download_wikitext_dataset(num_examples=1000):
    """
    Downloads a subset of the WikiText-103 dataset.
    
    Args:
        num_examples (int): Number of examples to download
        
    Returns:
        list[str]: List of text examples from WikiText
    """
    print(f"Downloading {num_examples} examples from WikiText-103...")
    try:
        # Use specific revision to avoid glob pattern issues
        dataset = datasets.load_dataset(
            "wikitext", 
            "wikitext-103-v1", 
            split="train", 
            revision="2b6e3d7c5a9cb2e1ee20eee7bc323d07a44c0c9f"
        )
        
        # Filter out empty lines and take only the specified number of examples
        filtered_texts = [text for text in dataset["text"] if len(text.strip()) > 50][:num_examples]
        
        print(f"Downloaded {len(filtered_texts)} examples from WikiText-103")
        return filtered_texts
    except Exception as e:
        print(f"Error downloading WikiText dataset: {e}")
        # Fallback to a simpler approach
        try:
            dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            filtered_texts = [text for text in dataset["text"] if len(text.strip()) > 50][:num_examples]
            print(f"Downloaded {len(filtered_texts)} examples from WikiText-103 (fallback method)")
            return filtered_texts
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            # Return some dummy text if all else fails
            print("Using dummy text as fallback")
            return ["This is a dummy text example."] * num_examples

def get_background_activations(
        model: HookedTransformer,
        extraction_layers: list[int],
        batch_size: int = 32,
        num_examples: int = 1000
) -> dict[int, torch.Tensor]:
    """
    Extract activations from a background dataset (WikiText) to compute mean activations.
    
    Args:
        model (HookedTransformer): The model to extract activations from
        extraction_layers (list[int]): The layers to extract activations from
        batch_size (int): Batch size for processing
        num_examples (int): Number of examples to use from WikiText
        
    Returns:
        dict[int, torch.Tensor]: Dictionary mapping layer indices to mean activations
    """
    # Download dataset
    texts = download_wikitext_dataset(num_examples)
    
    # Process in batches
    all_activations = {layer: [] for layer in extraction_layers}
    
    print("Extracting background activations...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Extract activations for this batch
        batch_activations = extract_activations_last_token(
            model, batch_texts, extraction_layers, batch_size
        )
        
        # Append to our collection
        for layer in extraction_layers:
            all_activations[layer].append(batch_activations[layer])
    
    # Concatenate and compute means
    mean_activations = {}
    for layer in extraction_layers:
        if all_activations[layer]:  # Check if we have any activations
            all_layer_activations = torch.cat(all_activations[layer], dim=0)
            mean_activations[layer] = torch.mean(all_layer_activations, dim=0).detach().cpu()
    
    print("Computed background mean activations")
    return mean_activations

def generate_hook_addition(
    steering_vector: torch.Tensor, 
    beta: float, 
    background_mean: torch.Tensor = None, 
    demean_scale: float = 1.0,
    demean_layer: int = None,
    current_layer: int = None,
    background_means_dict: dict = None,
    layer_specific_scales: dict = None
) -> Callable:
    """
    Generates a hook function to add a steering vector to the last token.
    If background_mean is provided, the steering vector will be demeaned by subtracting the background mean.

    Parameters:
    - steering_vector (torch.Tensor): Steering vector.
    - beta (float): Scaling factor for the final vector.
    - background_mean (torch.Tensor, optional): Mean activation from background dataset to subtract.
    - demean_scale (float): Scaling factor specifically for the demeaning component.
    - demean_layer (int, optional): Layer to use for demeaning (if different from current layer).
    - current_layer (int, optional): Current layer being processed.
    - background_means_dict (dict, optional): Dictionary of background means for all layers.
    - layer_specific_scales (dict, optional): Dictionary mapping layers to their specific demeaning scales.

    Returns:
    - function: Hook function for adding steering vector.
    """
    # Determine which background mean to use
    if demean_layer is not None and background_means_dict is not None and current_layer is not None:
        # Use a different layer's background mean for demeaning
        if demean_layer in background_means_dict:
            bg_mean = background_means_dict[demean_layer]
        else:
            print(f"Warning: Demean layer {demean_layer} not found in background_means_dict. Using None.")
            bg_mean = None
    else:
        # Use the provided background mean (traditional approach)
        bg_mean = background_mean
    
    # Determine which scale to use
    actual_scale = demean_scale
    if layer_specific_scales is not None and current_layer is not None and current_layer in layer_specific_scales:
        actual_scale = layer_specific_scales[current_layer]
    
    # If we have a background mean, demean the steering vector
    if bg_mean is not None:
        # Apply scaling to the demeaning component
        demeaned_vector = steering_vector - (bg_mean * actual_scale)
    else:
        demeaned_vector = steering_vector
        
    def last_token_steering_hook(resid_pre, hook):
        for i in range(resid_pre.shape[0]):
            current_token_index = resid_pre.shape[1] - 1
            resid_pre[i, current_token_index, :] += demeaned_vector.squeeze().to(resid_pre.device) * beta

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
    args = parser.parse_args()
    
    PATH_TO_DATA = "antonyms.json"
    MODEL_NAME = "EleutherAI/gpt-j-6b"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = "float16"
    N_SPLITS = 3
    N_PAIRS_PER_TRAINING_PROMPT = 10
    EXTRACTION_LAYERS = list(range(9,13))
    BETAS = [1,3,5]
    BATCH_SIZE = 32
    RESULTS_DIR = args.out_dir
    
    # Number of examples to use from WikiText for background statistics
    BACKGROUND_EXAMPLES = 500
    
    # Scaling factors for demeaning component
    DEMEAN_SCALES = [0.5, 1.0, 2.0]
    
    # For Run 3: Try different layers for demeaning
    # We'll use the best demean_scale from Run 2 (assuming it's 1.0 for now)
    BEST_DEMEAN_SCALE = 1.0
    
    # Define which layer combinations to try for demeaning
    # Format: (steering_layer, demean_layer)
    LAYER_COMBINATIONS = []
    for steering_layer in EXTRACTION_LAYERS:
        for demean_layer in EXTRACTION_LAYERS:
            if steering_layer != demean_layer:  # Only try cross-layer combinations
                LAYER_COMBINATIONS.append((steering_layer, demean_layer))
    
    # For Run 4: Layer-specific demeaning scales
    # Define different demeaning scales for each layer
    LAYER_SPECIFIC_SCALES = {}
    for layer in EXTRACTION_LAYERS:
        # Start with different scales for each layer based on layer depth
        # Deeper layers might need different scaling than earlier layers
        LAYER_SPECIFIC_SCALES[layer] = 0.5 + (layer / 20.0)  # Scales from ~0.5 to ~1.1 depending on layer

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # load data
    data = pd.read_json(PATH_TO_DATA)
    cv_pairs = get_cv_splits(data, n_splits=N_SPLITS)

    model = HookedTransformer.from_pretrained_no_processing(
        model_name=MODEL_NAME, device=DEVICE, dtype=DTYPE
    )
    model.eval()
    
    # Get background activations from WikiText
    background_means = get_background_activations(
        model, 
        EXTRACTION_LAYERS, 
        batch_size=BATCH_SIZE,
        num_examples=BACKGROUND_EXAMPLES
    )
    
    # Verify all required layers are in the background_means dictionary
    for layer in EXTRACTION_LAYERS:
        if layer not in background_means:
            print(f"Warning: Layer {layer} missing from background_means. Adding empty tensor.")
            background_means[layer] = torch.zeros(model.cfg.d_model, device=DEVICE)

    # create a dataframe to store the results
    results_df = pd.DataFrame(columns=['split', 'layer', 'beta', 'demean_scale', 'accuracy'])

    # Create a progress bar for the total number of iterations
    total_iterations = N_SPLITS * len(EXTRACTION_LAYERS) * len(BETAS) * len(DEMEAN_SCALES)
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
        
        # test steering vector on test prompts
        for layer in EXTRACTION_LAYERS:
            for beta in BETAS:
                for demean_scale in DEMEAN_SCALES:
                    # Use demeaned steering vector with different scaling factors
                    addition_hook = generate_hook_addition(
                        steering_vector=activations_last_token_averaged[layer], 
                        beta=beta,
                        background_mean=background_means[layer],
                        demean_scale=demean_scale
                    )
                    hooks = [(f"blocks.{layer}.hook_resid_pre", addition_hook)]
                    try:
                        predictions = generate_text(model, test_prompts, hooks)
                        accuracy = calculate_accuracy(predictions, data_test['target'])
                        print(f"Split {split_number}, Beta {beta}, Layer {layer}, Demean Scale {demean_scale}, Accuracy {accuracy}")
                    except Exception as e:
                        print(f"Error generating text for Split {split_number}, Beta {beta}, Layer {layer}, Demean Scale {demean_scale}: {e}")
                        predictions = ["ERROR"] * len(test_prompts)
                        accuracy = 0.0
                    
                    # Add to results dataframe
                    new_row = pd.DataFrame({
                        'split': [split_number], 
                        'layer': [layer], 
                        'beta': [beta],
                        'demean_scale': [demean_scale],
                        'accuracy': [accuracy]
                    })
                    
                    if len(results_df) == 0:
                        results_df = new_row
                    else:
                        results_df = pd.concat([results_df, new_row], ignore_index=True)
                
                    # Save predictions for this configuration
                    predictions_df = pd.DataFrame({
                        'prompt': test_prompts,
                        'prediction': predictions,
                        'target': data_test['target'],
                        'demean_scale': [demean_scale] * len(test_prompts)
                    })
                    predictions_df.to_csv(
                        os.path.join(RESULTS_DIR, f"predictions_{split_number}_{layer}_{beta}_{demean_scale}.csv"), 
                        index=False
                    )
                
                # save results after trying all demean_scales for this layer/beta combination
                results_df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)

                pbar.update(1)
    
    # Run the cross-layer demeaning experiment (Run 3)
    if LAYER_COMBINATIONS:
        print("\nStarting cross-layer demeaning experiment (Run 3)...")
        
        # Create a new progress bar for the cross-layer experiment
        cross_layer_iterations = N_SPLITS * len(LAYER_COMBINATIONS) * len(BETAS)
        cross_pbar = tqdm(total=cross_layer_iterations, desc="Cross-Layer Progress")
        
        # Create a new dataframe for cross-layer results
        cross_layer_results_df = pd.DataFrame(columns=[
            'split', 'steering_layer', 'demean_layer', 'beta', 'accuracy'
        ])
        
        for split_number in range(N_SPLITS):
            # Use the same data splits as before
            data_train, data_test = cv_pairs[split_number]
            training_prompts = generate_training_prompts(
                data_train, n_pairs_per_training_prompt=N_PAIRS_PER_TRAINING_PROMPT, separator=', ')
            test_prompts = generate_test_prompts(data_test)
            
            # Get activations for training prompts
            activations_last_token = extract_activations_last_token(
                model, training_prompts, EXTRACTION_LAYERS, batch_size=BATCH_SIZE)
            activations_last_token_averaged = average_activations(activations_last_token)
            
            # Try each layer combination
            for steering_layer, demean_layer in LAYER_COMBINATIONS:
                for beta in BETAS:
                    # Use the best demean scale from previous runs
                    addition_hook = generate_hook_addition(
                        steering_vector=activations_last_token_averaged[steering_layer],
                        beta=beta,
                        demean_layer=demean_layer,
                        current_layer=steering_layer,
                        background_means_dict=background_means,
                        demean_scale=BEST_DEMEAN_SCALE
                    )
                    
                    hooks = [(f"blocks.{steering_layer}.hook_resid_pre", addition_hook)]
                    try:
                        predictions = generate_text(model, test_prompts, hooks)
                        accuracy = calculate_accuracy(predictions, data_test['target'])
                        
                        print(f"Split {split_number}, Steering Layer {steering_layer}, " +
                              f"Demean Layer {demean_layer}, Beta {beta}, Accuracy {accuracy}")
                    except Exception as e:
                        print(f"Error in cross-layer experiment: Split {split_number}, Steering Layer {steering_layer}, " +
                              f"Demean Layer {demean_layer}, Beta {beta}: {e}")
                        predictions = ["ERROR"] * len(test_prompts)
                        accuracy = 0.0
                    
                    # Add to results dataframe
                    new_row = pd.DataFrame({
                        'split': [split_number],
                        'steering_layer': [steering_layer],
                        'demean_layer': [demean_layer],
                        'beta': [beta],
                        'accuracy': [accuracy]
                    })
                    
                    cross_layer_results_df = pd.concat([cross_layer_results_df, new_row], ignore_index=True)
                    
                    # Save predictions
                    predictions_df = pd.DataFrame({
                        'prompt': test_prompts,
                        'prediction': predictions,
                        'target': data_test['target'],
                        'steering_layer': [steering_layer] * len(test_prompts),
                        'demean_layer': [demean_layer] * len(test_prompts)
                    })
                    
                    predictions_df.to_csv(
                        os.path.join(RESULTS_DIR, 
                                    f"cross_predictions_{split_number}_{steering_layer}_{demean_layer}_{beta}.csv"),
                        index=False
                    )
                    
                    cross_pbar.update(1)
            
            # Save cross-layer results after each split
            cross_layer_results_df.to_csv(os.path.join(RESULTS_DIR, "cross_layer_results.csv"), index=False)
        
        cross_pbar.close()

    # Run 4: Layer-specific demeaning scales experiment
    if LAYER_SPECIFIC_SCALES:
        print("\nStarting layer-specific demeaning scales experiment (Run 4)...")
        
        # Create a new progress bar for the layer-specific experiment
        layer_specific_iterations = N_SPLITS * len(EXTRACTION_LAYERS) * len(BETAS)
        layer_specific_pbar = tqdm(total=layer_specific_iterations, desc="Layer-Specific Progress")
        
        # Create a new dataframe for layer-specific results
        layer_specific_results_df = pd.DataFrame(columns=[
            'split', 'layer', 'beta', 'layer_specific_scale', 'accuracy'
        ])
        
        for split_number in range(N_SPLITS):
            # Use the same data splits as before
            data_train, data_test = cv_pairs[split_number]
            training_prompts = generate_training_prompts(
                data_train, n_pairs_per_training_prompt=N_PAIRS_PER_TRAINING_PROMPT, separator=', ')
            test_prompts = generate_test_prompts(data_test)
            
            # Get activations for training prompts
            activations_last_token = extract_activations_last_token(
                model, training_prompts, EXTRACTION_LAYERS, batch_size=BATCH_SIZE)
            activations_last_token_averaged = average_activations(activations_last_token)
            
            # Try each layer with its specific demeaning scale
            for layer in EXTRACTION_LAYERS:
                for beta in BETAS:
                    # Use layer-specific demeaning scale
                    addition_hook = generate_hook_addition(
                        steering_vector=activations_last_token_averaged[layer],
                        beta=beta,
                        background_mean=background_means[layer],
                        current_layer=layer,
                        layer_specific_scales=LAYER_SPECIFIC_SCALES
                    )
                    
                    hooks = [(f"blocks.{layer}.hook_resid_pre", addition_hook)]
                    try:
                        predictions = generate_text(model, test_prompts, hooks)
                        accuracy = calculate_accuracy(predictions, data_test['target'])
                        
                        layer_specific_scale = LAYER_SPECIFIC_SCALES[layer]
                        print(f"Split {split_number}, Layer {layer}, Beta {beta}, " +
                              f"Layer-Specific Scale {layer_specific_scale:.2f}, Accuracy {accuracy}")
                    except Exception as e:
                        print(f"Error in layer-specific experiment: Split {split_number}, Layer {layer}, " +
                              f"Beta {beta}, Scale {LAYER_SPECIFIC_SCALES[layer]:.2f}: {e}")
                        predictions = ["ERROR"] * len(test_prompts)
                        accuracy = 0.0
                    
                    # Add to results dataframe
                    new_row = pd.DataFrame({
                        'split': [split_number],
                        'layer': [layer],
                        'beta': [beta],
                        'layer_specific_scale': [layer_specific_scale],
                        'accuracy': [accuracy]
                    })
                    
                    layer_specific_results_df = pd.concat([layer_specific_results_df, new_row], ignore_index=True)
                    
                    # Save predictions
                    predictions_df = pd.DataFrame({
                        'prompt': test_prompts,
                        'prediction': predictions,
                        'target': data_test['target'],
                        'layer': [layer] * len(test_prompts),
                        'layer_specific_scale': [layer_specific_scale] * len(test_prompts)
                    })
                    
                    predictions_df.to_csv(
                        os.path.join(RESULTS_DIR, 
                                    f"layer_specific_predictions_{split_number}_{layer}_{beta}.csv"),
                        index=False
                    )
                    
                    layer_specific_pbar.update(1)
            
            # Save layer-specific results after each split
            layer_specific_results_df.to_csv(os.path.join(RESULTS_DIR, "layer_specific_results.csv"), index=False)
        
        layer_specific_pbar.close()

    pbar.close()
    print("All experiments completed!")

