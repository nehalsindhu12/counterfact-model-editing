import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import seaborn as sns
import sys
import pickle
import os

device = torch.device("cuda:6")
print(f"Using device: {device}")

def get_paragraphs(text):
    paragraphs = text.split('\n\n')

    valid_paragraphs = []
    for p, para in enumerate(paragraphs):
        para = para.strip()
        if para.count('\n') == 0 and len(para) > 200:
            valid_paragraphs.append(para)

    return valid_paragraphs


def extract_layer_vectors(model, tokenizer, dataset, total_tokens = 100):
    layer_vectors = {}
    token_count = 0

    #create random order of articles
    article_indices = [i for i in range(len(dataset))]
    random.shuffle(article_indices)

    input_lengths = []
    selected_tokens = []
    for article_count, index in enumerate(article_indices):
        print(article_count, len(input_lengths))

        #select paragraph
        article_text = dataset[index]['text']
        valid_paragraphs = get_paragraphs(article_text)
        if len(valid_paragraphs) ==0:
            continue

        selected_paragraph = random.sample(valid_paragraphs, 1)
        paragraph = selected_paragraph[0]        

        #inputs = tokenizer(paragraph, return_tensors='pt', padding='max_length', truncation=True, max_length=1024)
        inputs = tokenizer(paragraph, return_tensors='pt', truncation=True, max_length=1024)
        article_len = torch.sum(inputs['input_ids'] != 50256).item()

        input_lengths.append(article_len)

        #move to device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            #select random token
            token_index = random.randint(0, input_lengths[-1] - 1)
            selected_tokens.append(token_index)
            for layer, hidden_state in enumerate(outputs.hidden_states):

                if layer not in layer_vectors:
                    layer_vectors[layer] = []
                vector = hidden_state[:, token_index, :].cpu().numpy().flatten()
                layer_vectors[layer].append(vector)  

        if len(input_lengths) > total_tokens:
            break

    return layer_vectors, input_lengths, selected_tokens



def pairwise_angles_different_contexts(layer_vectors, n_pairs=500):
    angles = []
    orthogonal_count = 0

    #sample pairs
    num_vectors = layer_vectors.shape[0]
    all_vectors = [i for i in range(num_vectors)]
    sampled_vectors = []
    for i in range(n_pairs):
        pair = random.sample(all_vectors,2)
        sampled_vectors.append(pair)

    for v_index, (v1, v2) in enumerate(sampled_vectors):
        print('Calculating pair', v_index)
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


if __name__ == '__main__':

    model_loc = "/data/akshat/models/pythia-6.9b"
    model_name = model_loc.split('/')[-1]
    n_paragraphs = int(1e4)
    n_pairs = int(1e5)

    tokenizer = AutoTokenizer.from_pretrained(model_loc)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_loc, output_hidden_states=True).to(device)
    dataset = load_dataset('wikipedia', '20220301.en', split='train')

    layer_vectors, input_lengths, selected_tokens = extract_layer_vectors(model, tokenizer, dataset, total_tokens = n_paragraphs)

    print(sys.getsizeof(layer_vectors)/ 2**20, 'MB')

    results = {}
    for layer in layer_vectors:
        vectors = np.vstack(layer_vectors[layer]) 
        if len(vectors) > 1: 
            mean_angle, std_dev, orthogonal_percentage = pairwise_angles_different_contexts(vectors, n_pairs= n_pairs)
            results[layer] = (mean_angle, std_dev, orthogonal_percentage)
        else:
            results[layer] = (None, None, None)
    
    print(sys.getsizeof(vectors) / 2**20, 'MB')

    layers = list(results.keys())
    mean_angles = [results[layer][0] if results[layer][0] is not None else 0 for layer in layers]
    std_devs = [results[layer][1] if results[layer][1] is not None else 0 for layer in layers]

    plt.figure(figsize=(10, 6))
    plt.errorbar(layers, mean_angles, yerr=std_devs, fmt='-o', capsize=5, label='Mean Angle with Std Dev')
    plt.title(f'Intercontext Pairwise Angles per Layer - {model_name}')
    plt.xlabel('Layer')
    plt.ylabel('Mean Angle (degrees)')
    plt.xticks(range(0, max(layers) + 1, 5))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.savefig(model_name + '_p' + str(n_paragraphs) + '_t' + str(n_pairs) + '_layer_angles_updated.png')
    plt.close()

    sns.histplot(input_lengths, bins=15, kde=True, color='purple')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Customized Distribution Plot")
    #plt.savefig(model_name + '_p' + str(n_paragraphs) + '_t' + str(n_pairs) + '_context_lengths.png')
    plt.close()

    sns.histplot(selected_tokens, bins=15, kde=True, color='purple')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Customized Distribution Plot")
    #plt.savefig(model_name + '_p' + str(n_paragraphs) + '_t' + str(n_pairs) + '_selected_tokens.png')
    plt.close()

    save_path = '/data/nehals/code/plotting-data/intercontext'

    with open(os.path.join(save_path, model_name + '_data.pkl'), 'wb') as file:
        pickle.dump({
            'layer_vectors': layer_vectors,
            'input_lengths': input_lengths,
            'selected_tokens': selected_tokens,
            'results': results
        }, file)


