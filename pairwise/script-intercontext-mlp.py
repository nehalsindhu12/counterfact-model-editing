import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import seaborn as sns
import json
import sys
import pickle
import os

os.environ["HF_DATASETS_CACHE"] = "/data/nehals/.cache/hf_datasets"
os.environ["TRANSFORMERS_CACHE"] = "/data/nehals/.cache/hf_transformers"
os.environ["XDG_CACHE_HOME"] = "/data/nehals/.cache"

device = torch.device("cuda:1")
print(f"Using device: {device}")

def get_paragraphs(text):
    paragraphs = text.split('\n\n')
    return [para.strip() for para in paragraphs if para.strip() and para.count('\n') == 0 and len(para) > 200]

def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

def register_hooks_for_all_layers(model, hook_layer_prefix, total_layers):
    global activations
    activations = [None] * total_layers 

    def capture_activation(layer_index):
        def hook(module, input, output):
            activations[layer_index] = input[0].detach()  
        return hook

    for layer_number in range(total_layers):
        layer_path = hook_layer_prefix.format(layer_number=layer_number)
        module = dict(model.named_modules()).get(layer_path, None)
        if module:
            module.register_forward_hook(capture_activation(layer_number))
            print(f"Hook registered at: {layer_path}")
        else:
            raise ValueError(f"Layer path '{layer_path}' not found in the model.")

def extract_layer_vectors(model, tokenizer, dataset, total_tokens):
    layer_vectors = {i: [] for i in range(len(activations))}  
    input_lengths = []
    selected_tokens = []

    article_indices = [i for i in range(len(dataset))]
    random.shuffle(article_indices)

    for article_count, index in enumerate(article_indices):
        print(article_count, len(input_lengths))

        article_text = dataset[index]['text']
        valid_paragraphs = get_paragraphs(article_text)
        if not valid_paragraphs:
            continue

        paragraph = random.choice(valid_paragraphs)
        inputs = tokenizer(paragraph, return_tensors='pt', truncation=True, max_length=1024)
        article_len = torch.sum(inputs['input_ids'] != tokenizer.pad_token_id).item()
        input_lengths.append(article_len)

        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            model(**inputs)

            token_index = random.randint(0, article_len - 1)
            selected_tokens.append(token_index)

            for layer_index, activation in enumerate(activations):
                if activation is not None:
                    vector = activation[:, token_index, :].cpu().numpy().flatten()
                    layer_vectors[layer_index].append(vector)

        if len(input_lengths) >= total_tokens:
            break

    return layer_vectors, input_lengths, selected_tokens

def pairwise_angles_different_contexts(layer_vectors, n_pairs):
    angles = []
    orthogonal_count = 0

    for layer in layer_vectors:
        vectors = np.vstack(layer_vectors[layer])
        num_vectors = len(vectors)
        if num_vectors < 2:
            continue

        sampled_indices = [random.sample(range(num_vectors), 2) for _ in range(min(n_pairs, num_vectors * (num_vectors - 1) // 2))]
        for v_index, (i, j) in enumerate(sampled_indices):
            print('Calculating pair', v_index)
            vec1, vec2 = vectors[i], vectors[j]
            cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            angles.append(angle)
            if 89 <= angle <= 91:
                orthogonal_count += 1

    mean_angle = np.mean(angles) if angles else 0
    std_dev = np.std(angles) if angles else 0
    percent_ortho = (orthogonal_count / len(angles)) * 100 if angles else 0

    return mean_angle, std_dev, percent_ortho

if __name__ == '__main__':
    config_path = 'config.json'
    config = load_config(config_path)
    model_key = 'pythia-6.9b' 
    model_config = config['models'][model_key]

    model_loc = model_config["model_path"]
    hook_layer_prefix = model_config["hook_layer_prefix"]
    total_layers = model_config["total_layers"]
    n_paragraphs = int(1e4)
    n_pairs = int(1e5)

    tokenizer = AutoTokenizer.from_pretrained(model_loc)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_loc, output_hidden_states=True).to(device)
    print(model)

    register_hooks_for_all_layers(model, hook_layer_prefix, total_layers)

    dataset = load_dataset('wikipedia', '20220301.en', split='train')
    layer_vectors, input_lengths, selected_tokens = extract_layer_vectors(model, tokenizer, dataset, total_tokens=n_paragraphs)

    results = {}
    for layer in layer_vectors:
        vectors = np.vstack(layer_vectors[layer])
        if len(vectors) > 1:
            mean_angle, std_dev, orthogonal_percentage = pairwise_angles_different_contexts({layer: layer_vectors[layer]}, n_pairs)
            results[layer] = (mean_angle, std_dev, orthogonal_percentage)
        else:
            results[layer] = (None, None, None)

    layers = list(results.keys())
    mean_angles = [results[layer][0] if results[layer][0] is not None else 0 for layer in layers]
    std_devs = [results[layer][1] if results[layer][1] is not None else 0 for layer in layers]

    plt.figure(figsize=(10, 6))
    plt.errorbar(layers, mean_angles, yerr=std_devs, fmt='-o', capsize=5, label='Mean Angle with Std Dev')
    plt.title('Intercontext Pairwise Angles per Layer - ' + model_key.upper())
    plt.xlabel('Layer')
    plt.ylabel('Mean Angle (degrees)')
    plt.xticks(range(0, total_layers, 5))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.savefig(model_key + '_p' + str(n_paragraphs) + '_t' + str(n_pairs) + '_mlp_angles.png')
    plt.close()

    sns.histplot(input_lengths, bins=15, kde=True, color='purple')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Paragraph Lengths")
    #plt.savefig(model_key + '_p' + str(n_paragraphs) + '_t' + str(n_pairs) + 'mlp_context_lengths.png')
    plt.close()

    sns.histplot(selected_tokens, bins=15, kde=True, color='purple')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Selected Token Positions")
    #plt.savefig(model_key + '_p' + str(n_paragraphs) + '_t' + str(n_pairs) + 'mlp_selected_tokens.png')
    plt.close()


    save_path = '/data/nehals/code/plotting-data/intercontext-mlp'

    with open(os.path.join(save_path, model_key + '_mlp.pkl'), 'wb') as file:
        pickle.dump({
            'layer_vectors': layer_vectors,
            'input_lengths': input_lengths,
            'selected_tokens': selected_tokens,
            'results': results
        }, file)