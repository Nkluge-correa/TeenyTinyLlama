import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import SentencePieceBPETokenizer
from transformers import LlamaTokenizerFast, TrainingArguments, AutoTokenizer

def main(args):

    # Load the dataset from the huggingface Hub and prepare it for training.
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, 
            split=args.dataset_split, 
            token=args.hub_token if args.hub_token else None,
        )
    else:
        raise ValueError("No dataset name provided or dataset is already tokenized") 

    # Remove non text columns.
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

    # Select `num_samples` from the dataset
    # According to our tests, 2 million samples is enough to train a good tokenizer without running into memory issues.
    dataset = dataset.shuffle(seed=42).select(range(arg.num_samples))

    # Create a SentencePieceBPETokenizer.
    tokenizer = SentencePieceBPETokenizer()

    # Train the SentencePieceBPETokenizer on the dataset.
    tokenizer.train_from_iterator(
        iterator=dataset['text'],
        vocab_size=args.vocab_size,
        show_progress=True,
        special_tokens=["<unk>", "<s>", "</s>",  "<pad>"],
    )

    # Save the tokenizer.
    tokenizer.save("new-sentencepiece-tokenizer.json", pretty=True)

    # Load reference tokenizer.
    if args.reference_tokenizer is not None and args.hub_token is not None:
        reference_tokenizer = AutoTokenizer.from_pretrained(args.reference_tokenizer, token=args.hub_token if args.hub_token else None)
        reference_tokenizer.save_pretrained("reference-tokenizer")
    else:
        raise ValueError("No tokenizer name provided or no hub token provided. Try using `--reference_tokenizer 'meta-llama/Llama-2-7b-hf'")

    # Read and dump the json file for the new tokenizer and the reference tokenizer.
    with open("new-sentencepiece-tokenizer.json") as f:
        new_llama_tokenizer_json = json.load(f)

    with open("reference-tokenizer/tokenizer.json") as f:
        reference_tokenizer_json = json.load(f)
    
    # Add the reference tokenizer's config to the new tokenizer's config.
    new_llama_tokenizer_json["normalizer"] = reference_tokenizer_json["normalizer"]
    new_llama_tokenizer_json["pre_tokenizer"] = reference_tokenizer_json["pre_tokenizer"]
    new_llama_tokenizer_json["post_processor"] = reference_tokenizer_json["post_processor"]
    new_llama_tokenizer_json["decoder"] = reference_tokenizer_json["decoder"]
    new_llama_tokenizer_json["model"]['fuse_unk'] = reference_tokenizer_json["model"]['fuse_unk']
    new_llama_tokenizer_json["model"]['byte_fallback'] = reference_tokenizer_json["model"]['byte_fallback']

    # Dump the new tokenizer's config.
    with open("new-sentencepiece-tokenizer.json", "w") as f:
        json.dump(new_llama_tokenizer_json, f, indent=2, ensure_ascii=False)

    # Load the new tokenizer as a LlamaTokenizerFast.
    new_llama_tokenizer = LlamaTokenizerFast(
        tokenizer_file="new-sentencepiece-tokenizer.json",
        unk_token="<unk>",
        unk_token_id=0,
        bos_token="<s>",
        bos_token_id=1,
        eos_token="</s>",
        eos_token_id=2,
        pad_token="<pad>",
        pad_token_id=3,
        padding_side="right",
    )

    # Save the new tokenizer.
    new_llama_tokenizer.save_pretrained("new-llama-tokenizer")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a new Llama tokenizer")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to be tokenized",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="The split of the dataset to be tokenized",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to access the dataset on the hub",
    )
    parser.add_argument(
        "--reference_tokenizer",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="The name of the reference tokenizer to use",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2_000_000,
        help="Number of samples to use from the dataset. You might run into memory issues if you use too many samples.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size to use for the tokenizer",
    )
    args = parser.parse_args()
    main(args)

# How to run:
# python train-sentencepiece.py --dataset_name "nicholasKluge/Pt-Corpus-Instruct" --dataset_split "train" --hub_token "hf_..." --reference_tokenizer "meta-llama/Llama-2-7b-hf" --num_samples 2000000 --vocab_size 32000