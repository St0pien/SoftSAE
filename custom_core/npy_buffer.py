import numpy as np
import torch as t
import gc


class NpyActivationBuffer:
    """
    Activation buffer backed by a memory-mapped .npy file.

    Compatible with the sampling semantics of ActivationBuffer:
    - Random batches
    - Refresh when < 50% unread
    - Supports activation norm filtering
    """

    def __init__(
        self,
        npy_path: str,
        npy_length: int,
        d_submodule: int,
        n_ctxs=3e4,
        ctx_len=128,
        out_batch_size=8192,
        device="cpu",
        dtype=t.float16,
        max_activation_norm_multiple: int | None = None,
        seed: int | None = None,
    ):
        self.npy_path = npy_path
        self.npy_length = npy_length
        self.d_submodule = d_submodule
        self.n_ctxs = int(n_ctxs)
        self.ctx_len = int(ctx_len)
        self.activation_buffer_size = self.n_ctxs * self.ctx_len
        self.out_batch_size = out_batch_size
        self.device = device
        self.dtype = dtype
        self.remove_high_norm = max_activation_norm_multiple

        if seed is not None:
            np.random.seed(seed)
            t.manual_seed(seed)

        # Memory-map the file (no load into RAM)
        self.mm = np.memmap(self.npy_path, shape=(npy_length, d_submodule))
        assert self.mm.ndim == 2, "Expected shape (N, d_submodule)"
        assert (
            self.mm.shape[1] == d_submodule
        ), f"d_submodule mismatch: file has {self.mm.shape[1]}"

        self.file_size = self.mm.shape[0]
        self.file_idx = 0  # pointer into memmap

        # Buffer tensors
        self.activations = t.empty(
            0, d_submodule, device=device, dtype=dtype
        )
        self.read = t.zeros(0, dtype=t.bool, device=device)

        # Initial fill
        self.refresh()

    def __iter__(self):
        return self

    def __next__(self):
        with t.no_grad():
            # Refresh if buffer < 50% unread
            if (~self.read).sum() < self.activation_buffer_size // 2:
                self.refresh()

            unreads = (~self.read).nonzero(as_tuple=False).squeeze(1)
            if len(unreads) == 0:
                raise StopIteration

            idxs = unreads[
                t.randperm(len(unreads), device=self.device)[
                    : self.out_batch_size
                ]
            ]
            self.read[idxs] = True
            return self.activations[idxs]

    def refresh(self):
        gc.collect()
        t.cuda.empty_cache()

        # Keep unread activations
        if len(self.activations) > 0:
            self.activations = self.activations[~self.read]

        current_idx = len(self.activations)

        # Allocate new buffer
        new_acts = t.empty(
            self.activation_buffer_size,
            self.d_submodule,
            device=self.device,
            dtype=self.dtype,
        )
        new_acts[:current_idx] = self.activations
        self.activations = new_acts

        # Fill from file
        while current_idx < self.activation_buffer_size:
            remaining = self.activation_buffer_size - current_idx
            to_read = min(remaining, self.file_size)

            # Wrap around file if needed
            end = self.file_idx + to_read
            if end <= self.file_size:
                chunk = self.mm[self.file_idx:end]
                self.file_idx = end
            else:
                part1 = self.mm[self.file_idx :]
                part2 = self.mm[: end - self.file_size]
                chunk = np.concatenate([part1, part2], axis=0)
                self.file_idx = end - self.file_size

            chunk = t.from_numpy(chunk).to(
                device=self.device, dtype=self.dtype
            )

            # Optional high-norm filtering
            if self.remove_high_norm is not None:
                norms = chunk.norm(dim=-1)
                median = norms.median()
                mask = norms <= median * self.remove_high_norm
                chunk = chunk[mask]

            if len(chunk) == 0:
                continue

            chunk = chunk[:remaining]
            self.activations[current_idx : current_idx + len(chunk)] = chunk
            current_idx += len(chunk)

        self.read = t.zeros(
            len(self.activations), dtype=t.bool, device=self.device
        )

    @property
    def config(self):
        return {
            "d_submodule": self.d_submodule,
            "n_ctxs": self.n_ctxs,
            "ctx_len": self.ctx_len,
            "out_batch_size": self.out_batch_size,
            "device": self.device,
            "npy_path": self.npy_path,
            "npy_length": self.npy_length,
        }
