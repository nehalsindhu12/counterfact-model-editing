import json
import numpy as np
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random

device = torch.device("cuda:0")

model_loc = "/data/akshat/models/gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_loc, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_loc, output_hidden_states=True, local_files_only=True).to(device)

counterfact_file = "/data/maochuanlu/counterfact-model-editing/counterfact/data/counterfact.json"
with open(counterfact_file, "r") as f:
    counterfact_dataset = json.load(f)

save_dir = "/data/maochuanlu/counterfact-model-editing/orthogonal_fact_selection_cf_format/"
os.makedirs(save_dir, exist_ok=True)

log_file = os.path.join(save_dir, "orthogonality_log_1111.json")

#store global activations from each layer of gpt2-xl, need to change if model is different
global activations
activations = [None] * len(model.transformer.h)


def capture_activation(layer_index):
    def hook(module, input, output):
        activations[layer_index] = input[0].detach().clone()
    return hook

for layer_index, layer in enumerate(model.transformer.h):
    layer.mlp.c_proj.register_forward_hook(capture_activation(layer_index))

def find_subject_token_index(sentence_tokens, subject, tokenizer):
    # reconstruct the full sentence from tokens
    detokenized_sentence = tokenizer.convert_tokens_to_string(sentence_tokens)
    # return The starting character index where the first occurrence of subject begins in this sentence.
    start_pos = detokenized_sentence.find(subject)

    # If it doesn't appear, return the last token index:
    if start_pos == -1:
        return None#len(sentence_tokens) - 1  

    # walk through tokens one by one to find which token contains the end of the subject
    char_count = 0
    for token_idx, token in enumerate(sentence_tokens):
        token_str = tokenizer.convert_tokens_to_string([token])
        char_count += len(token_str)
        if char_count > start_pos + len(subject) - 1:
            return token_idx

    return len(sentence_tokens) - 1  

def get_activation_vector(sentence, subject):
    # Converts the sentence into token IDs
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(device)

    # gives the token strings so I can locate where the subject appears
    sentence_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
    last_subject_token_index = find_subject_token_index(sentence_tokens, subject, tokenizer)

    with torch.no_grad():
        model(**inputs)

    selected_layer = 17  
    # selects the activation at that subject token position
    vector = activations[selected_layer][:, last_subject_token_index, :].cpu().numpy().flatten()
    return vector

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

#random 10 numbers: 111, 222, 333, 444, 555, 666, 777, 888, 999, 1111
reference_fact = counterfact_dataset[1111]  
print(f"Reference fact prompt: {reference_fact['requested_rewrite']['prompt']}")
print(f"Reference fact subject: {reference_fact['requested_rewrite']['subject']}")
# Reference fact prompt: {}, who holds a citizenship from
# Reference fact subject: Armin Hofmann

#fill up sentence
reference_sentence = reference_fact["requested_rewrite"]["prompt"].format(reference_fact["requested_rewrite"]["subject"])

#get the activation vector for the subject token in this sentence
reference_vector = get_activation_vector(reference_sentence, reference_fact["requested_rewrite"]["subject"])

fact_vectors = []
for fact in tqdm(counterfact_dataset):
    sentence = fact["requested_rewrite"]["prompt"].format(fact["requested_rewrite"]["subject"])
    vector = get_activation_vector(sentence, fact["requested_rewrite"]["subject"])
    fact_vectors.append((fact, vector))

fact_angles = []
for fact, vector in fact_vectors:
    similarity = cosine_similarity(reference_vector, vector)
    # covert the angle in degrees
    angle = np.degrees(np.arccos(np.clip(similarity, -1.0, 1.0)))
    #if angle > 0: 
    fact_angles.append((fact, angle))

log_data = [{"fact": fact, "angle": angle} for fact, angle in fact_angles]
with open(log_file, "w") as f:
    json.dump(log_data, f, indent=4)
print(f"Logged orthogonality scores to {log_file}")

filtered_facts = [entry for entry in fact_angles if 89 <= entry[1] <= 91]
filtered_facts.sort(key=lambda x: abs(x[1] - 90))
most_orthogonal = [fact for fact, angle in filtered_facts[:10]]

fact_angles.sort(key=lambda x: x[1])
least_orthogonal = [fact for fact, angle in fact_angles[:10]] 

most_orthogonal_file = os.path.join(save_dir, "most_orthogonal_1111.json")
least_orthogonal_file = os.path.join(save_dir, "least_orthogonal_1111.json")

with open(most_orthogonal_file, "w") as f:
    json.dump(most_orthogonal, f, indent=4)
with open(least_orthogonal_file, "w") as f:
    json.dump(least_orthogonal, f, indent=4)

print(f"Saved most orthogonal dataset to {most_orthogonal_file}")
print(f"Saved least orthogonal dataset to {least_orthogonal_file}")
