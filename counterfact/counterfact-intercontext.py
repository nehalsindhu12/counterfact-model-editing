import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from counterfact import CounterFactDataset
import seaborn as sns
import pickle
import os
import sys

device = torch.device("cuda:1")
print(f"Using device: {device}")

def find_subject_token_index(sentence_tokens, subject, tokenizer):
    detokenized_sentence = tokenizer.convert_tokens_to_string(sentence_tokens)
    
    start_pos = detokenized_sentence.find(subject)

    if start_pos == -1:
        return len(sentence_tokens) - 1  

    char_count = 0
    for token_idx, token in enumerate(sentence_tokens):
        token_str = tokenizer.convert_tokens_to_string([token])
        char_count += len(token_str)
        if char_count > start_pos + len(subject) - 1:
            return token_idx

    return len(sentence_tokens) - 1  

def extract_layer_vectors(model, tokenizer, dataset, total_tokens=100):
    layer_vectors = {}
    input_lengths = []
    selected_tokens = []

    debug_output = []  
    for article_count, example in enumerate(dataset):
        subject = example['requested_rewrite']['subject']
        prompt = example['requested_rewrite']['prompt']
        sentence = prompt.format(subject)

        sentence_tokens = tokenizer.convert_ids_to_tokens(
            tokenizer(sentence, return_tensors="pt")['input_ids'][0].tolist()
        )

        last_subject_token_index = find_subject_token_index(sentence_tokens, subject, tokenizer)

        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=1024)
        article_len = torch.sum(inputs['input_ids'] != tokenizer.pad_token_id).item()
        input_lengths.append(article_len)
        selected_tokens.append(last_subject_token_index)

        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            for layer, hidden_state in enumerate(outputs.hidden_states):
                if layer not in layer_vectors:
                    layer_vectors[layer] = []
                vector = hidden_state[:, last_subject_token_index, :].cpu().numpy().flatten()
                layer_vectors[layer].append(vector)

        reconstructed_token = tokenizer.decode([inputs['input_ids'][0, last_subject_token_index]]).strip()
        correct_match = subject in sentence and reconstructed_token in sentence


        debug_output.append({
            "Example Number": article_count,
            "Sentence": sentence,
            "Sentence Tokens": sentence_tokens,
            "Subject": subject,
            "Token at Last Subject Index": sentence_tokens[last_subject_token_index],
            "Reconstructed Token": reconstructed_token,
            "Correct Match": correct_match
        })

        if len(input_lengths) >= total_tokens:
            break

    with open("debug_output.txt", "w") as debug_file:
        for entry in debug_output:
            debug_file.write(f"{entry}\n")

    return layer_vectors, input_lengths, selected_tokens

def pairwise_angles_different_contexts(layer_vectors, n_pairs=500):
    angles = []
    orthogonal_count = 0

    num_vectors = layer_vectors.shape[0]
    sampled_vectors = [random.sample(range(num_vectors), 2) for _ in range(n_pairs)]

    for v_index, (v1, v2) in enumerate(sampled_vectors):
        vec1, vec2 = layer_vectors[v1], layer_vectors[v2]
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        angles.append(angle)
        if 89 <= angle <= 91:
            orthogonal_count += 1

    mean_angle = np.mean(angles) if angles else 0
    std_dev = np.std(angles) if angles else 0
    percent_ortho = (orthogonal_count / len(sampled_vectors)) * 100 if sampled_vectors else 0

    return mean_angle, std_dev, percent_ortho

if __name__ == "__main__":
    model_loc = "/data/akshat/models/Llama-2-7b-hf" 
    model_name = model_loc.split("/")[-1]
    n_paragraphs = int(1e4)
    n_pairs = int(1e5)

    tokenizer = AutoTokenizer.from_pretrained(model_loc)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_loc, output_hidden_states=True).to(device)

    data_dir = "/data/nehals/code/counterfact/data"
    counterfact_dataset = CounterFactDataset(data_dir)

    layer_vectors, input_lengths, selected_tokens = extract_layer_vectors(
        model, tokenizer, counterfact_dataset, total_tokens=n_paragraphs
    )

    results = {}
    for layer in layer_vectors:
        vectors = np.vstack(layer_vectors[layer])
        if len(vectors) > 1:
            mean_angle, std_dev, orthogonal_percentage = pairwise_angles_different_contexts(vectors, n_pairs)
            results[layer] = (mean_angle, std_dev, orthogonal_percentage)
        else:
            results[layer] = (None, None, None)

    layers = list(results.keys())
    mean_angles = [results[layer][0] if results[layer][0] is not None else 0 for layer in layers]
    std_devs = [results[layer][1] if results[layer][1] is not None else 0 for layer in layers]

    # plt.figure(figsize=(10, 6))
    # plt.errorbar(layers, mean_angles, yerr=std_devs, fmt='-o', capsize=5, label='Mean Angle with Std Dev')
    # plt.title(f'Intercontext Pairwise Angles per Layer - {model_name}')
    # plt.xlabel('Layer')
    # plt.ylabel('Mean Angle (degrees)')
    # plt.xticks(range(0, max(layers) + 1, 5))
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(model_name + '_p' + str(n_paragraphs) + '_t' + str(n_pairs) + '_cf_layer_angles_updated.png')
    # plt.close()

    # sns.histplot(input_lengths, bins=15, kde=True, color='purple')
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # plt.title("Customized Distribution Plot")
    # plt.savefig(model_name + '_p' + str(n_paragraphs) + '_t' + str(n_pairs) + '_cf_context_lengths.png')
    # plt.close()

    # sns.histplot(selected_tokens, bins=15, kde=True, color='purple')
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # plt.title("Customized Distribution Plot")
    # plt.savefig(model_name + '_p' + str(n_paragraphs) + '_t' + str(n_pairs) + '_cf_selected_tokens.png')
    # plt.close()

    # save_path = '/data/nehals/code/plotting-data'
    # with open(os.path.join(save_path, model_name + 'counterfact_data.pkl'), 'wb') as file:
    #     pickle.dump({
    #         'layer_vectors': layer_vectors,
    #         'input_lengths': input_lengths,
    #         'selected_tokens': selected_tokens,
    #         'results': results
    #     }, file)
