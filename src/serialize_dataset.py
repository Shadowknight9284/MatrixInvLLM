import numpy as np

def serialize_matrix(matrix, precision=5, value_separator=", "):
    """
    Serialize a numeric matrix into a CSV-style text string.
    """
    rows = []
    for row in matrix:
        formatted_row = value_separator.join([f"{val:.{precision}f}" for val in row])
        rows.append(formatted_row)
    return "\n".join(rows)

def create_prompt_target(example, config):
    """
    Construct the input_text and target_text strings based on config settings.
    """
    ser_cfg = config.get("serialization", {})
    precision = ser_cfg.get("fixed_decimal_precision", 5)
    val_sep = ser_cfg.get("value_separator", ", ")
    
    # Headers
    input_header = ser_cfg.get("input_header", "Matrix")
    output_header = ser_cfg.get("output_header", "Inverse")
    
    n = example["matrix_size"]
    
    # Build Input Text
    input_lines = []
    if ser_cfg.get("include_matrix_size", True):
        input_lines.append(f"MatrixSize: {n}")
    
    input_lines.append(f"{input_header}:")
    input_lines.append(serialize_matrix(example["matrix"], precision, val_sep))
    
    input_text = "\n".join(input_lines)
    
    # Build Target Text
    target_lines = []
    target_lines.append(f"{output_header}:")
    target_lines.append(serialize_matrix(example["inverse"], precision, val_sep))
    
    target_text = "\n".join(target_lines)
    
    return input_text, target_text

def apply_serialization_to_dataset(dataset, config):
    """
    Add input_text and target_text to each example in the dataset.
    """
    for example in dataset:
        input_text, target_text = create_prompt_target(example, config)
        example["input_text"] = input_text
        example["target_text"] = target_text
    return dataset
