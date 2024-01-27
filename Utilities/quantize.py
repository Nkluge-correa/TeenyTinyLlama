import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo
from transformers import AwqConfig, AutoConfig

def main(args):
    # Load the target model
    model = AutoAWQForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Quantize the model
    quant_config = {
        "zero_point": True, 
        "q_group_size": 128, 
        "w_bit": 4, 
        "version":"GEMM"
    }
    model.quantize(tokenizer, quant_config=quant_config)

    # Modify the model config to include the quantization config
    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()

    # Add the quantization config to the model config
    model.model.config.quantization_config = quantization_config

    # Save model and tokenizer
    model.save_quantized(args.quant_path)
    tokenizer.save_pretrained(args.quant_path)

    # Upload to the Hub
    api = HfApi(token=args.token)

    create_repo(
    repo_id=f"{args.model_path}-awq", 
    token=args.token,
    repo_type="model",
    exist_ok=True,
    private=True)

    api.upload_folder(
        folder_path=args.quant_path,
        repo_id=f"{args.model_path}-awq",
        repo_type="model",
    )

    print(f"Model saved to {args.model_path}-awq!")

if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Quantize a model.')
    parser.add_argument('--token', type=str, required=True, help='Hugging Face token')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--quant_path', type=str, required=True, help='Path to save the quantized model')
    args = parser.parse_args()

    main(args)

# How to run this script:
# python quantize.py --token "hf_..." --model_path "nicholasKluge/TeenyTinyLlama-460m" --quant_path "TeenyTinyLlama-460m-awq"