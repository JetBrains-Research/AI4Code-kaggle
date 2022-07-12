# taken from https://github.com/agemagician/CodeTrans
import json
import os
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import SummarizationPipeline, AutoModelWithLMHead, AutoTokenizer

from kendall_tau import compute_kendall_tau


def comment_perplexity(pipeline, code, comment):
    code_tokens = pipeline.tokenizer.tokenize(code)
    code_ids = pipeline.tokenizer.convert_tokens_to_ids(code_tokens)

    comment_tokens = pipeline.tokenizer.tokenize(comment)
    comment_ids = pipeline.tokenizer.convert_tokens_to_ids(comment_tokens)

    outputs = pipeline.model(torch.tensor([code_ids]), labels=torch.tensor([comment_ids]))
    log_like = outputs.logits[0, range(len(comment_ids)), comment_ids].sum()

    return log_like


def sequence_perplexity_batched(pipeline, code_ids_tensor, comments_ids_tensor, comments_ids_mask, comment_index,
                                token_index):
    comment_n = comments_ids_tensor.size()[0]
    outputs = pipeline.model(code_ids_tensor.tile(comment_n, 1), labels=comments_ids_tensor)

    logits = outputs.logits[comment_index, token_index, comments_ids_tensor]
    logits_masked = torch.mul(logits, comments_ids_mask)
    return torch.sum(logits_masked, dim=1)


def compute_cost_mtx(code_cells_input, comment_cells_input):
    cost_mtx = []
    for code_cell in tqdm(code_cells_input):
        code_cell_costs = []
        for comment_cell in comment_cells_input:
            log_likelihood = comment_perplexity(
                pipeline,
                train_ex["source"][code_cell][:64],
                train_ex["source"][comment_cell][:64]
            )
            code_cell_costs.append(float(log_likelihood))
        cost_mtx.append(code_cell_costs)
    return cost_mtx


def compute_cost_mtx_batched(code_ids_tensor, comments_ids_tensor, comments_ids_mask, comment_index, token_index):
    cost_mtx = []
    for code_cell_ids in tqdm(code_ids_tensor):
        code_cell_costs = sequence_perplexity_batched(
            pipeline,
            code_cell_ids,
            comments_ids_tensor,
            comments_ids_mask,
            comment_index,
            token_index
        )
        cost_mtx.append(code_cell_costs.tolist())
    return cost_mtx


def compute_cost_mtx_batched_batched(
        code_ids_tensor,
        comments_ids_tensor,
        comments_ids_mask,
        comment_index,
        comment_token_index
):
    code_n, comment_n = code_ids_tensor.size()[0], comments_ids_tensor.size()[0]
    outputs = pipeline.model(
        code_ids_tensor.repeat(1, comment_n).reshape(code_n * comment_n, code_ids_tensor.size()[1]),
        labels=comments_ids_tensor.tile(code_n, 1)
    )

    logits = outputs.logits[
        torch.tensor([[j for i in range(comments_ids_tensor.size()[1])] for j in range(comment_n * code_n)]),
        comment_token_index.tile(code_n, 1),
        comments_ids_tensor.tile(code_n, 1)
    ]
    logits_masked = torch.mul(logits, comments_ids_mask.tile(code_n, 1))
    # print(outputs.logits.size())

    cost_mtx = torch.sum(logits_masked, dim=1).reshape(code_n, comment_n)
    return cost_mtx


def prepare_comments(comments):
    comments_ids, n_comment_tokens = [], []
    for comment in comments:
        comment_tokens = pipeline.tokenizer.tokenize(comment)
        comment_ids = pipeline.tokenizer.convert_tokens_to_ids(comment_tokens)
        comments_ids.append(comment_ids)
        n_comment_tokens.append(len(comment_ids))
    max_comment_tokens = max(n_comment_tokens)

    comments_ids_tensor = torch.zeros((len(comments), max_comment_tokens), dtype=torch.long).to(torch.device("cpu"))
    comments_ids_mask = torch.zeros((len(comments), max_comment_tokens)).to(torch.device("cpu"))
    for i, comment_ids in enumerate(comments_ids):
        comments_ids_tensor[i, :len(comment_ids)] = torch.tensor(comment_ids)
        comments_ids_mask[i, :len(comment_ids)] = 1

    n_comments = comments_ids_tensor.size(0)
    comment_index = torch.tensor(range(n_comments)).repeat((max_comment_tokens, 1)).T
    token_index = torch.tensor(range(max_comment_tokens)).tile((n_comments, 1))
    return comments_ids_tensor, comments_ids_mask, comment_index, token_index


pipeline = SummarizationPipeline(
    model=AutoModelWithLMHead.from_pretrained("SEBIS/code_trans_t5_small_code_documentation_generation_python"),
    tokenizer=AutoTokenizer.from_pretrained("SEBIS/code_trans_t5_small_code_documentation_generation_python",
                                            skip_special_tokens=True),
    device=torch.device("cpu")
)

# code = "def e(message, exit_code=None):\n   print_log(message, YELLOW, BOLD)\n" \
#        "    if exit_code is not None:\n        sys.exit(exit_code)"  # @param {type:"raw"}
#
# comment = "<pad> Prints an error message and exits with a default exit code.</s>"
#
# log_likelihood = comment_perplexity(pipeline, code, comment)
# print("log-likelihood-2", log_likelihood)
# #log-likelihood-2 tensor(223.5326, grad_fn=<SumBackward0>)

train_orders = pd.read_csv("../../AI4Code/train_orders.csv")
json_list_train = os.listdir("../../AI4Code/train")

train_ex = json_list_train[0]
train_ex_id = train_ex[:-5]
print(train_ex_id)
print(train_orders[train_orders.id == train_ex_id])

with open(f"../../AI4Code/train/{train_ex_id}.json") as file:
    train_ex = json.load(file)

code_cells = [k for k, v in train_ex["cell_type"].items() if v == "code"]
comment_cells = [k for k, v in train_ex["cell_type"].items() if v == "markdown"]

code_cells_values_len = [len(train_ex["source"][k]) for k in code_cells]
comment_cells_values_len = [len(train_ex["source"][k]) for k in comment_cells]
print(max(comment_cells_values_len), max(code_cells_values_len))
print(sum(comment_cells_values_len) / len(comment_cells), sum(code_cells_values_len) / len(code_cells))

code_cells_values = [train_ex["source"][k][:32] for k in code_cells]
comment_cells_values = [train_ex["source"][k][:32] for k in comment_cells]

comments_ids_tensor, comments_ids_mask, comment_index, comment_token_index = prepare_comments(comment_cells_values)
code_ids_tensor, *_ = prepare_comments(code_cells_values)

correct_assignment = train_orders[train_orders.id == train_ex_id].cell_order.values[0].split(" ")

start = time.time()
cost_mtx = compute_cost_mtx_batched_batched(
    code_ids_tensor,
    comments_ids_tensor,
    comments_ids_mask,
    comment_index,
    comment_token_index
).detach().numpy()
# print(cost_mtx)
print(time.time() - start)
kendall_tau_score = compute_kendall_tau(cost_mtx, comment_cells, code_cells, correct_assignment)
print("kendall_tau_score", kendall_tau_score)

start = time.time()
cost_mtx = compute_cost_mtx_batched(
    code_ids_tensor,
    comments_ids_tensor,
    comments_ids_mask,
    comment_index,
    comment_token_index
)
# print(cost_mtx)
print(time.time() - start)
kendall_tau_score = compute_kendall_tau(cost_mtx, comment_cells, code_cells, correct_assignment)
print("kendall_tau_score", kendall_tau_score)

cost_mtx = compute_cost_mtx(code_cells, comment_cells)
print(cost_mtx)
kendall_tau_score = compute_kendall_tau(cost_mtx, comment_cells, code_cells, correct_assignment)
print("kendall_tau_score", kendall_tau_score)
