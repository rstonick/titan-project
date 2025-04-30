import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import re # Added for layer filtering

# Remove quantization_type from arguments, read from config instead
def prepare_model(config):
    #\"\"\"Prepare the model with quantization and LoRA configuration\"\"\"

    # Read quantization type from config
    quantization_type = config.get('quantization', {}).get('quantization_type', 'none') # Default to 'none' if not specified

    # Quantization configuration
    if quantization_type == '4bit':
        print("Applying 4-bit quantization...") # Added print
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=config['quantization'].get('use_double_quant', False), # Use .get for safety
            bnb_4bit_quant_type=config['quantization'].get('quant_type', 'nf4'), # Default to nf4 for 4bit
            bnb_4bit_compute_dtype=getattr(torch, config['quantization'].get('compute_dtype', 'float16')), # Default compute dtype
        )
    elif quantization_type == '8bit':
        print("Applying 8-bit quantization...") # Added print
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # 8bit specific parameters (if any, often fewer than 4bit)
            # Example: bnb_8bit_compute_dtype might be relevant if supported
            # Check bitsandbytes documentation for relevant 8bit args in BitsAndBytesConfig
        )
    else:
        print("No quantization applied.") # Added print
        bnb_config = None

    # Determine torch_dtype based on quantization or config default
    if bnb_config:
        # If quantizing, compute_dtype is often handled by bnb_config, but set torch_dtype for non-quantized layers
        model_dtype = getattr(torch, config.get('quantization', {}).get('compute_dtype', 'float16'))
    else:
        # If no quantization, use a default like float16 or read from another config field if available
        model_dtype = torch.float16 # Or read from a general model config if desired

    # Load model with quantization if specified
    print(f"Loading model {config['model']['name']} with dtype {model_dtype}...") # Added print
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2", # Added Flash Attention 2
        torch_dtype=model_dtype, # Use determined dtype
        # device_map='auto' # Removed or commented out for accelerate compatibility
    )
    print("Model loaded.") # Added print

    # Prepare for k-bit training only if quantized
    if quantization_type in ['4bit', '8bit']:
        print(f"Preparing model for {quantization_type} training...") # Added print
        # Ensure gradient checkpointing is enabled for memory saving with k-bit training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) # Pass model directly, enable GC
        print("Model prepared for k-bit training.") # Added print
    else:
        # If not quantized, you might still want gradient checkpointing
        # Check if model has 'gradient_checkpointing_enable' method
        if hasattr(model, 'gradient_checkpointing_enable'):
             print("Enabling gradient checkpointing for non-quantized model...")
             model.gradient_checkpointing_enable()


    # --- Dynamic Target Module Generation ---
    lora_config_data = config['lora']
    target_modules_config = lora_config_data['target_modules']
    target_layer_range = lora_config_data.get('target_layer_range') # Get the range if it exists

    final_target_modules = []
    if target_layer_range and isinstance(target_layer_range, list) and len(target_layer_range) == 2:
        start_layer, end_layer = target_layer_range
        # Assuming Mistral-like layer naming convention: model.layers.{index}.{module_type}.{proj_name}
        # Adjust the pattern if your model uses a different naming scheme
        layer_pattern = re.compile(r"model\.layers\.(\d+)\..*")
        print(f"Filtering LoRA target modules for layers {start_layer} to {end_layer}") # Added print

        # Find all linear layers potentially targetable by LoRA
        # This is a simplified approach; a more robust method might inspect module types
        all_linear_layers = set()
        for name, module in model.named_modules():
             # Check if the module name corresponds to one of the base target types (e.g., gate_proj)
             # AND if it falls within the specified layer range
             is_target_type = any(base_module in name for base_module in target_modules_config)
             if is_target_type:
                 match = layer_pattern.match(name)
                 if match:
                     layer_idx = int(match.group(1))
                     if start_layer <= layer_idx <= end_layer:
                         # Add the specific layer name (e.g., model.layers.10.mlp.gate_proj)
                         # We add the base module name found in the full name
                         for base_module in target_modules_config:
                             if base_module in name:
                                 final_target_modules.append(name) # Add the full name found
                                 break # Avoid adding duplicates if name contains multiple base modules

        # Remove duplicates if any module name matched multiple base targets
        final_target_modules = sorted(list(set(final_target_modules)))

        if not final_target_modules:
             print(f"Warning: No target modules found matching {target_modules_config} within layer range {target_layer_range}. Check module names and range.")
        else:
             print(f"Applying LoRA to {len(final_target_modules)} modules in layers {start_layer}-{end_layer}.")
             # print("Targeted modules:", final_target_modules) # Optional: print the full list

    else:
        # Fallback to using the provided target_modules list directly
        final_target_modules = target_modules_config
        print(f"Applying LoRA to specified target_modules: {final_target_modules}")


    # Create LoRA configuration using the dynamically determined or fallback list
    lora_config = LoraConfig(
        r=lora_config_data['basic']['r'],
        lora_alpha=lora_config_data['basic']['alpha'],
        target_modules=final_target_modules, # Use the generated/fallback list
        lora_dropout=lora_config_data['basic']['dropout'],
        bias=lora_config_data['advanced']['bias'],
        task_type=lora_config_data['advanced']['task_type'],
        fan_in_fan_out=lora_config_data['advanced']['fan_in_fan_out'],
        modules_to_save=lora_config_data['advanced']['modules_to_save'],
        init_lora_weights=lora_config_data['advanced']['init_lora_weights'],
        rank_pattern=lora_config_data['rank_pattern'],
        alpha_pattern=lora_config_data['alpha_pattern'],
    )
    # --- End Dynamic Target Module Generation ---


    # Apply LoRA to model
    try:
        model = get_peft_model(model, lora_config)
    except ValueError as e:
        print(f"Error applying PEFT model: {e}")
        print("Please check if the final_target_modules list is empty or contains invalid module names.")
        print("Final target modules attempted:", final_target_modules)
        raise e


    # Print trainable parameters
    model.print_trainable_parameters()

    return model