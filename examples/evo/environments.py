import jax
import jax.numpy as jnp

import numpy as np

from datasets import load_dataset

def safe_decode(tokens, tokenizer):
    try:
        stop_tokens = np.flatnonzero(tokens==0)
        if stop_tokens.size > 0:
            tokens = tokens[:stop_tokens[0]]
        return tokenizer.decode(tokens)
    except BaseException as e:
        return ""

class BaseTask:
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        self.encoding_tokenizer = encoding_tokenizer
        self.decoding_tokenizer = decoding_tokenizer
        self.max_num_steps = max_num_steps

    def __len__(self):
        """
        Get number of tasks or -1 if infinite
        """
        raise NotImplementedError("len not implemented")

    def get_input(self, indices):
        """
        Get the inputs corresponding to the indices (provided as a jax array, size N).
        Output is a jax matrix of size NxT, where 0s are tokens to be filled in by the model.
        The output should follow the same sharding as the indices
        """
        raise NotImplementedError("get_input not implemented")

    def get_batch_fitness(self, indices, full_generations):
        """
        Get the fitness for the generations (NxT) given indices (N)
        Output is a jax matrix of floats of size N
        """
        raise NotImplementedError("get_batch_fitness not implemented")


class ToyTask(BaseTask):

    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps, single_fitness):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps)
        self._batch_fitness = jax.jit(jax.vmap(single_fitness))

    def __len__(self):
        return -1

    def get_input(self, indices):
        print("task shape", indices.shape)
        return jnp.zeros(indices.shape + (self.max_num_steps,), dtype=jnp.int32, device=indices.device)

    def get_batch_fitness(self, indices, full_generations):
        print("task shape", indices.shape, full_generations.shape, indices.device, full_generations.device)
        return jax.device_put(self._batch_fitness(full_generations[:, 1:]), full_generations.device) # skip 0 token


class FastZero(ToyTask):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps,
                         lambda generated_tokens: -jax.numpy.nonzero(generated_tokens == 0, size=1, fill_value=generated_tokens.shape[0]*2)[0][0].astype(jnp.float32)
                         )

class UniqueTok(ToyTask):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps,
                         lambda generated_tokens: jnp.sum(jnp.where(jnp.unique_counts(generated_tokens, size=generated_tokens.shape[0]).counts == 0, 0, 1)).astype(jnp.float32)
                         )

class RepTok(ToyTask):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps,
                         lambda generated_tokens: jnp.max(jnp.unique_counts(generated_tokens, size=generated_tokens.shape[0]).counts).astype(jnp.float32)
                         )

class Digits(BaseTask):
    def __len__(self):
        return -1

    def get_input(self, indices):
        return jnp.zeros(indices.shape + (self.max_num_steps,), dtype=jnp.int32, device=indices.device)

    def get_batch_fitness(self, indices, full_generations):
        numpy_tokens = np.array(full_generations)
        num_digits = [sum(c.isdigit() for c in safe_decode(numpy_tokens[i, 1:], self.decoding_tokenizer)) for i in range(numpy_tokens.shape[0])]
        return jnp.array(num_digits, dtype=jnp.float32, device=full_generations.device)


def make_conversation(example):
    # return {"prompt": SYSTEM_PROMPT + "\n\n" + f"User: {example['question']}" + "\n\nAssistant: <think"}
    return {"prompt": f"User: {example['question']}\n\nAssistant: <think"}


import re

# code from https://github.com/tianlwang/eval_gsm8k

def extract_predicted_answer(text):
    regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore =[
        ",",
        "\\$",
        "(?s).*#### ",
        "\\.$"
    ]
    match = re.findall(regex_pattern, text)
    if match:
        match = match[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        text = match.strip()

        for regex in regexes_to_ignore:
            text = re.sub(regex, "", text)
        return text
    else:
        print("NO REGEX MATCH FOUND")
        return None

def extract_ground_truth(text):
    return text.split('####')[-1].strip()

def check_accuracy(generated_ans, solution):
    ground_truth_answer = extract_ground_truth(solution)
    # print(f"ground truth answer: {ground_truth_answer}")
    # print("model answer (unparsed)", generated_ans.strip())
    model_answer = extract_predicted_answer(generated_ans.strip())
    # print(f"model answer: {model_answer}; ground truth answer: {ground_truth_answer}")
    return 1.0 if (model_answer == ground_truth_answer) else 0.0
    
def single_fitness(generated_answer, true_answer, i):
    find_idx = generated_answer.find("</think>")
    if find_idx == -1:
        return 0.0
    true_idx = find_idx + len("</think>")
    generated_ans = generated_answer[true_idx:]
    return check_accuracy(generated_answer[true_idx:], true_answer)
    
def get_padded_prompt(single_prompt, generation_length):
    single_prompt = single_prompt[:generation_length]
    return single_prompt + [0] * (generation_length - len(single_prompt))

class GSM8KTrain(BaseTask):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        self.encoding_tokenizer = encoding_tokenizer
        self.decoding_tokenizer = decoding_tokenizer
        self.max_num_steps = max_num_steps

        self.dataset = load_dataset("openai/gsm8k", "main", split="train")
        self.dataset = self.dataset.map(make_conversation)

    def __len__(self):
        return len(self.dataset)

    def get_input(self, indices):
        # print([self.dataset[i % len(self.dataset)] for i in indices])
        return jnp.array([get_padded_prompt(self.encoding_tokenizer.encode(self.dataset[i.item() % len(self.dataset)]["prompt"]), self.max_num_steps) for i in indices], device=indices.device)


    def get_batch_fitness(self, indices, full_generations):
        """
        Get the fitness for the generations (NxT) given indices (N)
        Output is a jax matrix of floats of size N
        """
        batch_answers = [self.dataset[i.item() % len(self.dataset)]["answer"] for i in indices]
        rewards = []
        np_full_generations = np.array(full_generations)
        saw_correct = False
        saw_incorrect = False
        for i, tok_seq in enumerate(np_full_generations):
            gen_ans = safe_decode(tok_seq, self.decoding_tokenizer)
            if len(gen_ans) == 0:
                reward = 0.0
            else:
                reward = single_fitness(gen_ans, batch_answers[i], i)
            rewards.append(reward)
            # if reward == 0.0 and not saw_incorrect:
            #     print("Incorrect sample:", i)
            #     print("*"*20)
            #     print(gen_ans)
            #     print("*"*20)
            #     # print(tok_seq)
            #     saw_incorrect=True

            # if reward == 1.0 and not saw_correct:
            #     print("Correct sample:", i)
            #     print("*"*20)
            #     print(gen_ans)
            #     print("*"*20)
            #     saw_correct=True
        return jnp.array(rewards, dtype=jnp.float32, device=full_generations.device)

class GSM8KTest(GSM8KTrain):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        self.encoding_tokenizer = encoding_tokenizer
        self.decoding_tokenizer = decoding_tokenizer
        self.max_num_steps = max_num_steps

        self.dataset = load_dataset("openai/gsm8k", "main", split="test").map(make_conversation)

all_tasks = {
    "fastzero": FastZero,
    "uniquetok": UniqueTok,
    "reptok": RepTok,
    "digits": Digits,
    "gsm8k": GSM8KTrain
}

validation_tasks = {
    "fastzero": FastZero,
    "uniquetok": UniqueTok,
    "reptok": RepTok,
    "digits": Digits,
    "gsm8k": GSM8KTest
}
