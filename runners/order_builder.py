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
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        inv = OrderBuilder._count_inversions(ranks)
        n = len(gt)
        max_inv = n * (n - 1)
        return inv, max_inv
