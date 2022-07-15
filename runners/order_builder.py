from collections import defaultdict
from bisect import bisect

import numpy as np
import torch


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
