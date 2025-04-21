import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def prepare_model(config, quantization_type='4bit'):
    """Prepare the model with quantization and LoRA configuration"""
    
    # Quantization configuration
    if quantization_type == '4bit':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=config['quantization']['use_double_quant'],
            bnb_4bit_quant_type=config['quantization']['quant_type'],
            bnb_4bit_compute_dtype=getattr(torch, config['quantization']['compute_dtype']),
        )
    elif quantization_type == '8bit':
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=config['quantization']['use_double_quant'],
            bnb_8bit_quant_type=config['quantization']['quant_type'],
            bnb_8bit_compute_dtype=getattr(torch, config['quantization']['compute_dtype']),
        )
    else:
        bnb_config = None

    # Load model with quantization if specified
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for 4-bit training
    model = prepare_model_for_kbit_training(model)

    # Create LoRA configuration
    lora_config = LoraConfig(
        r=config['lora']['basic']['r'],
        lora_alpha=config['lora']['basic']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['basic']['dropout'],
        bias=config['lora']['advanced']['bias'],
        task_type=config['lora']['advanced']['task_type'],
        fan_in_fan_out=config['lora']['advanced']['fan_in_fan_out'],
        modules_to_save=config['lora']['advanced']['modules_to_save'],
        init_lora_weights=config['lora']['advanced']['init_lora_weights'],
        rank_pattern=config['lora']['rank_pattern'],
        alpha_pattern=config['lora']['alpha_pattern'],
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model 