import json
import json_repair
import logging
from logging.handlers import RotatingFileHandler
import os
import pickle
import re
from pathlib import Path

# Function to convert float32 to float
def convert_float32(obj):
    return float(obj)

def cache_output(output, file_name):
    if file_name.endswith(".txt"):
        ## store GPT4 output into a txt file
        with open(file_name, "w") as f:
            f.write(output)
    elif file_name.endswith(".json"):
        ## store GPT4 output into a json file
        with open(file_name, "w") as f:
            json.dump(output, f, indent=4)
    return

def get_content_between_a_b(start_tag, end_tag, text):
    extracted_text = ""
    start_index = text.find(start_tag)
    # print(text[start_index:start_index+100])
    # find the last one as the output
    while start_index != -1:
        next_start_index = text.find(start_tag, start_index + len(start_tag))
        if next_start_index != -1:
            start_index = next_start_index
            # print("find next_start_index:", text[start_index:start_index+100])
        else:
            break
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break

    return extracted_text.strip()


def extract(text, type,hard = True):
    if text:
        target_str = get_content_between_a_b(f"<{type}>", f"</{type}>", text)
        if target_str:
            return target_str
        elif hard:
            return text
        else:
            return ""
    else:
        return ""


def extract_json(text):
    if "```json" in text:
        target_str = get_content_between_a_b("```json", "```", text)
        return target_str
    else:
        return text

# Function to extract JSON string between specific markers from a given text
def extract_json_between_markers(llm_output, json_start_marker = "```json", json_end_marker = "```"):
    # Find the start and end indices of the JSON string
    start_index = llm_output.find(json_start_marker)
    if start_index != -1:
        start_index += len(json_start_marker)  # Move past the marker
        end_index = llm_output.find(json_end_marker, start_index)
    else:
        return None  # JSON markers not found

    if end_index == -1:
        return None  # End marker not found

    # Extract the JSON string
    json_string = llm_output[start_index:end_index].strip()
    try:
        parsed_json = json_repair.loads(json_string)
        return parsed_json
    except json.JSONDecodeError:
        return None  # Invalid JSON format

# Function to create a directory if it does not exist
def safe_mkdir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

# Function to load data from a pickle file
def load_pickle_from_file(pickle_file):
    with open(pickle_file, 'rb') as fr:
        return pickle.load(fr)

# Function to save data to a pickle file
def save_pickle_data_to_file(pickle_data, output_pickle_file):
    with open(output_pickle_file, 'wb') as fw:
        pickle.dump(pickle_data, fw)

# Function to load data from a JSON file
def load_json_from_file(json_file):
    with open(json_file, 'r') as fr:
        return json.load(fr)

# Function to save data to a JSON file
def save_json_data_to_file(json_data, output_json_file):
    with open(output_json_file, 'w') as fw:
        json.dump(json_data, fw, ensure_ascii=False, indent=4)

# Function to initialize logging configuration
def init_logging(log_dir, log_filename="app.log", max_log_size=1024*1024*5, backup_count=5):
    # If the log directory does not exist, create it
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Define the full path of the log file
    log_file_path = os.path.join(log_dir, log_filename)

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level

    # Create a handler for writing log to a file
    file_handler = RotatingFileHandler(log_file_path, maxBytes=max_log_size, backupCount=backup_count)
    file_handler.setLevel(logging.INFO)  # Set the handler level

    # Create a handler for outputting log to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set the handler level

    # Define the output format for the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def normalize_title(title):
    # Remove special characters and replace with spaces
    normalized_title = re.sub(r'[^\w\s]', '_', title)
    # Convert to lowercase
    normalized_title = normalized_title.lower()
    # Replace multiple spaces with a single space
    normalized_title = re.sub(r'\s+', '_', normalized_title)
    # Remove leading and trailing spaces
    normalized_title = normalized_title.strip()
    return normalized_title

def sanitize_filename(filename, replacement="_"):
    """Clean up a single filename component."""
    # Replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', replacement, filename)
    sanitized = sanitized.strip(" .")
    if not sanitized:
        sanitized = "unnamed"
    # Truncate to 255 bytes (UTF-8)
    while len(sanitized.encode('utf-8')) > 255:
        sanitized = sanitized[:-1]
    return sanitized

def normalize_path(path_str, max_total_length=4096):
    """Normalize an entire path string."""
    parts = []
    for part in Path(path_str).parts:
        parts.append(sanitize_filename(part))
    normalized = Path(*parts).resolve()
    normalized_str = str(normalized)

    # Optional: truncate total path length
    if len(normalized_str.encode('utf-8')) > max_total_length:
        print(f"Warning: Path exceeds {max_total_length} bytes. Truncating...")

        parent_parts = list(normalized.parent.parts)
        name = normalized.name

        new_parent_parts = []
        for p in parent_parts:
            while len(p.encode('utf-8')) > 200:
                p = p[:-1]
            new_parent_parts.append(p)

        normalized_str = str(Path(*new_parent_parts) / name)

    return normalized_str

if __name__ == '__main__':
    # Example usage
    # Initialize logging configuration
    log_directory = "log"
    init_logging(log_directory)

    # In other modules, just import the logging module and get the root logger
    import logging
    logger = logging.getLogger()

    # Log messages
    logger.info("This is an info message")
    logger.error("This is an error message")


    print(normalize_title("AI4Research: Use LLM to generate novel research ideas"))
    print(normalize_title("gui agent"))