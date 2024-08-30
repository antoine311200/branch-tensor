from __future__ import annotations

import torch
from opt_einsum import contract

from typing import Union

class BranchTensor:

    def __init__(self, input_shape, output_shape, ranks):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.ranks = ranks

        self.branches = []

    def decompose(self, tensor: torch.Tensor, level: Union[str, int] = "full"):
        """Decompose a tensor into a binary tree of branch tensors with level pruning
        
        Recursively decompose a tensor using a series of QR decomposition, storing the top rows
        of the Q matrix as the major core and the bottom rows as the minor core. The same thing is
        done with the R matrix, storing the left columns as the major remainder and the right columns
        as the minor remainder. The process is repeated recursively on the remaining core until the desired
        level of pruning is reached.

        Args:
            tensor (torch.Tensor): The tensor to decompose
            level (Union[str, int], optional): The level of pruning. Defaults to "full".
        """
        if level == "full":
            level = len(self.input_shape) - 1

        def branch_decompose(tensor, input_shape, output_shape, ranks, level):
            if len(input_shape) == 1:
                return [
                    (tensor.reshape(ranks[0], input_shape[0], output_shape[0], -1), None),
                    (None, None)
                ]

            # Check if empty tensor
            if tensor.numel() == 0:
                return None   

            left_unfolded = tensor.reshape(ranks[0] * input_shape[0] * output_shape[0], -1)

            core, remainder = torch.linalg.qr(left_unfolded, mode='complete')

            core_major = core[:, :ranks[1]]
            core_minor = core[:, ranks[1]:2*ranks[1]]

            remainder_major = remainder[:ranks[1], :]
            remainder_minor = remainder[ranks[1]:2*ranks[1], :]

            core_major = core_major.reshape(ranks[0], input_shape[0], output_shape[0], ranks[1])
            core_minor = core_minor.reshape(ranks[0], input_shape[0], output_shape[0], -1)#remainder.shape[0] - ranks[1])

            ranks_major = ranks[1:]
            ranks_minor = [core_minor.shape[-1]] + ranks[2:]

            return [
                (core_major, branch_decompose(remainder_major, input_shape[1:], output_shape[1:], ranks_major, level)),
                (core_minor, branch_decompose(remainder_minor, input_shape[1:], output_shape[1:], ranks_minor, level - 1) if level != 0 else None)
            ]

        self.branches = branch_decompose(tensor, self.input_shape, self.output_shape, self.ranks, level)

        return self

    def recompose(self):
        """Recompose the tensor from the binary tree of branch tensors"""
        
        def branch_recompose(tree):
            if tree is None or tree == (None, None):
                return None
            
            if tree[0][0] is not None and tree[0][1] is None:
                return tree[0][0]

            core_major, branches_major = tree[0]
            core_minor, branches_minor = tree[1]

            # Left branch processing

            left_einsum = []
            left_compose = branch_recompose(branches_major)

            left_shape = tuple(f"axis_{i}" for i in range(len(left_compose.shape) - 1))

            left_einsum.append(core_major)
            left_einsum.append(("left", "input", "output", "right"))
            left_einsum.append(left_compose)
            left_einsum.append(("right", ) + left_shape)
            left_einsum.append(("left", "input", "output") + left_shape)

            # Right branch processing

            right_einsum = []
            right_compose = branch_recompose(branches_minor)

            if right_compose is None:
                return contract(*left_einsum).squeeze()
            
            right_shape = tuple(f"axis_{i}" for i in range(len(right_compose.shape) - 1))

            right_einsum.append(core_minor)
            right_einsum.append(("left", "input", "output", "right"))
            right_einsum.append(right_compose)
            right_einsum.append(("right", ) + right_shape)
            right_einsum.append(("left", "input", "output") + right_shape)

            return contract(*left_einsum).squeeze() + contract(*right_einsum).squeeze()

        return branch_recompose(self.branches)

    def prune(self, level):
        def prune(tree, level):
            if tree is None or tree == (None, None):
                return None
            
            core_major, branches_major = tree[0]
            core_minor, branches_minor = tree[1]

            if level == 0:
                return [(core_major, prune(branches_major, 0)), (None, None)]

            return [(core_major, prune(branches_major, level)), (core_minor, prune(branches_minor, level - 1))]
        
        self.branches = prune(self.branches, level)
    
    def __repr__(self) -> str:
        repr_str = f"BranchTensor(input_shape={self.input_shape}, output_shape={self.output_shape}, ranks={self.ranks})"
    
        def print_tree(tree, indent=0):
            if tree is None or tree == (None, None):
                return""
            for core, branches in tree:
                if core is None:
                    continue

                return " " * indent+"↪" + core.shape + "\n" + print_tree(branches, indent + 3)
        
        return repr_str+"\n"+print_tree(self.branches)

    def __eq__(self, other: BranchTensor) -> bool:
        def equal(tree1, tree2):
            if tree1 is None and tree2 is None:
                return True
            if tree1 is None or tree2 is None:
                return False

            if tree1[0][0] is not None and tree2[0][0] is not None:
                return torch.allclose(tree1[0][0], tree2[0][0])

            return equal(tree1[0][1], tree2[0][1]) and equal(tree1[1][1], tree2[1][1])

        return equal(self.branches, other.branches)