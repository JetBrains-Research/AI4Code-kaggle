from collections import defaultdict
from bisect import bisect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import json


class OrderBuilder:
    @staticmethod
    def greedy_pairwise(probs: torch.Tensor, is_code):
        code_positions = np.where(is_code)[0]
        next_cells = defaultdict(list)

        for i in range(len(is_code)):
            if not is_code[i]:
                for position in torch.argsort(probs[:, i], descending=True):
                    if is_code[position]:
                        next_cells[position].append(i)
                        break
                else:
                    raise ValueError("No code cells in notebook?")

        order = []
        for position in code_positions:
            order.append(position)
            order.extend(next_cells[position])

        return order

    @staticmethod
    def greedy_pairwise_reverse(probs: torch.Tensor, is_code):
        code_positions = np.where(is_code)[0]
        prev_cells = defaultdict(list)

        for i in range(len(is_code)):
            if not is_code[i]:
                for position in torch.argsort(probs[i, :], descending=True):
                    position = position.cpu().item()
                    if is_code[position]:
                        prev_cells[position].append(i)
                        break
                else:
                    raise ValueError("No code cells in notebook?")

        order = []
        for position in reversed(code_positions):
            order.append(position)
            order.extend(prev_cells[position])

        order = list(reversed(order))
        return order

    @staticmethod
    def greedy_ranked(positions, n_md, n_code):
        n = n_md + n_code
        order = [-1] * n
        positions = sorted(list(enumerate(positions)), key=lambda x: x[1])

        cur_code_ind = 0
        cur_md_ind = 0
        for i in range(n):
            if (cur_code_ind >= n_code) or (cur_md_ind < n_md and positions[cur_md_ind][1] <= i):
                order[i] = n_code + positions[cur_md_ind][0]
                cur_md_ind += 1
            else:
                order[i] = cur_code_ind
                cur_code_ind += 1

        return order

    @staticmethod
    def greedy_ranked_reversed(positions, n_md, n_code):
        n = n_md + n_code
        order = [-1] * n
        positions = sorted(list(enumerate(positions)), key=lambda x: x[1])

        cur_code_ind = n_code - 1
        cur_md_ind = n_md - 1
        for i in reversed(range(n)):
            if (cur_code_ind < 0) or (cur_md_ind >= 0 and positions[cur_md_ind][1] >= i):
                order[i] = n_code + positions[cur_md_ind][0]
                cur_md_ind -= 1
            else:
                order[i] = cur_code_ind
                cur_code_ind -= 1

        return order

    @staticmethod
    def kaggle_ranker(rel_positions, n_md, n_code):
        n = n_md + n_code
        ranks = [
            (i, (i + 1) / (n_code + 1))
            for i in range(n_code)
        ] + [
            (n_code + i, pos)
            for i, pos in enumerate(rel_positions)
        ]
        ranks.sort(key=lambda x: x[1])
        return [x[0] for x in ranks]

    @staticmethod
    def _count_inversions(a):
        inversions = 0
        sorted_so_far = []
        for i, u in enumerate(a):  # O(N)
            j = bisect(sorted_so_far, u)  # O(log N)
            inversions += i - j
            sorted_so_far.insert(j, u)  # O(N)
        return inversions

    @staticmethod
    def kendall_tau(gt, pred):
        ranks = [
            gt.index(x) for x in pred
        ]  # rank predicted order in terms of ground truth
        inv = OrderBuilder._count_inversions(ranks)
        n = len(gt)
        max_inv = n * (n - 1)
        return inv, max_inv

    @staticmethod
    def get_json(notebook_id):
        notebook_json = notebook_id + ".json"
        p = Path("../raw_data/train/") / notebook_json
        d = json.load(open(p, 'r'))
        return d

    @staticmethod
    def get_cell_types(notebook_id):
        return OrderBuilder.get_json(notebook_id)['cell_type']

    @staticmethod
    def evaluate_notebooks(notebooks_order):
        orders = pd.read_csv("../raw_data/train_orders.csv").set_index("id")
        total_inv = 0
        total_max_inv = 0
        for notebook, pred_order in notebooks_order.items():
            true_order = orders.cell_order[notebook].split()

            cell_types = OrderBuilder.get_cell_types(notebook)
            code_cells = [cell_id for cell_id in true_order if cell_types[cell_id] == "code"]

            for i, pred in enumerate(pred_order):
                if isinstance(pred, str):
                    continue
                pred_order[i] = code_cells[pred]

            inv, max_inv = OrderBuilder.kendall_tau(true_order, pred_order)
            total_inv += inv
            total_max_inv += max_inv
        return 1 - 4 * total_inv / total_max_inv
