from bisect import bisect

from scipy.optimize import linear_sum_assignment
import numpy as np


def count_inversions(a):  # Actually O(N^2), but fast in practice for our data
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):  # O(N)
        j = bisect(sorted_so_far, u)  # O(log N)
        inversions += i - j
        sorted_so_far.insert(j, u)  # O(N)
    return inversions


def kendall_tau(ground_truth, predictions):
    total_inversions = 0  # total inversions in predicted ranks across all instances
    total_2max = 0  # maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


def compute_kendall_tau(cost_mtx, comment_cell_ids, code_cell_ids, correct_assignment):
    row_ind, col_ind = linear_sum_assignment(np.array(cost_mtx))
    assignment = [[comment_cell_ids[j], code_cell_ids[i]] for i, j in zip(row_ind, col_ind)]
    assignment_flat = [x for xs in assignment for x in xs]
    my_assignment = [correct_assignment.index(cell_id) for cell_id in assignment_flat]
    correct_assignment_ind = list(range(len(correct_assignment)))
    return kendall_tau([correct_assignment_ind], [my_assignment])
