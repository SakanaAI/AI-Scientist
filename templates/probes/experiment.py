import pandas as pd
from sklearn.model_selection import KFold
import torch
from transformer_lens import HookedTransformer
from typing import Callable
from tqdm import tqdm
import os
import argparse
import json

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

def generate_hook_addition(steering_vector: torch.Tensor, beta: float) -> Callable:
    """
    Generates a hook function to add a steering vector to the last token.

    Parameters:
    - steering_vector (torch.Tensor): Steering vector.
    - beta (float): Scaling factor.

    Returns:
    - function: Hook function for adding steering vector.
    """
    def last_token_steering_hook(resid_pre, hook):
        for i in range(resid_pre.shape[0]):
            current_token_index = resid_pre.shape[1] - 1
            resid_pre[i, current_token_index, :] += steering_vector.squeeze().to(resid_pre.device) * beta

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
    parser = argparse.ArgumentParser(description="Run experiments with demeaned probes")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Directory to save results")
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

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # load dat
    data = pd.read_json(PATH_TO_DATA)
    cv_pairs = get_cv_splits(data, n_splits=N_SPLITS)

    model = HookedTransformer.from_pretrained_no_processing(
        model_name=MODEL_NAME, device=DEVICE, dtype=DTYPE
    )
    model.eval()

    # create a dataframe to store the results
    results_df = pd.DataFrame(columns=['split', 'layer', 'beta', 'accuracy'])

    # Create a progress bar for the total number of iterations
    total_iterations = N_SPLITS * len(EXTRACTION_LAYERS) * len(BETAS)
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
                addition_hook = generate_hook_addition(steering_vector=activations_last_token_averaged[layer], beta=beta)
                hooks = [(f"blocks.{layer}.hook_resid_pre", addition_hook)]
                predictions = generate_text(model, test_prompts, hooks)
                accuracy = calculate_accuracy(predictions, data_test['target'])
                print(f"Split {split_number}, Beta {beta}, Layer {layer}, Accuracy {accuracy}")
                if len(results_df) == 0:
                    results_df = pd.DataFrame({
                        'split': [split_number], 
                        'layer': [layer], 
                        'beta': [beta], 
                        'accuracy': [accuracy]
                    })
                else:
                    results_df = pd.concat([results_df, pd.DataFrame({
                        'split': [split_number], 
                        'layer': [layer], 
                        'beta': [beta], 
                        'accuracy': [accuracy]
                    })], ignore_index=True)
                
                # save results
                results_df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)

                predictions_df = pd.DataFrame({
                    'prompt': test_prompts,
                    'prediction': predictions,
                    'target': data_test['target']
                })
                predictions_df.to_csv(os.path.join(RESULTS_DIR, f"predictions_{split_number}_{layer}_{beta}.csv"), index=False)

                pbar.update(1)
    
    pbar.close()
    print("All experiments completed!")

    # Compute and save averaged results using groupby
    grouped_results = results_df.groupby(['layer', 'beta'])['accuracy'].agg(['mean', 'std']).reset_index()
    averaged_results = {
        f"layer_{row['layer']}_beta_{row['beta']}": {
            "means": float(row['mean']),
            "stds": float(row['std'])
        }
        for _, row in grouped_results.iterrows()
    }
    
    # Save to final_info.json
    with open(os.path.join(RESULTS_DIR, "final_info.json"), "w") as f:
        json.dump(averaged_results, f, indent=4)

