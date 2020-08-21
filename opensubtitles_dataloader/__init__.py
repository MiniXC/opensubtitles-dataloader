__version__ = "0.1.3"

from torch.utils.data import Dataset, random_split
from tqdm.auto import tqdm
import os.path as path
import os
import mmap
from collections import deque
import numpy as np
import torch


class OpenSubtitlesDataset(Dataset):
    """Open subtitles dataset,
    see http://opus.nlpl.eu/OpenSubtitles-v2018.php"""

    def __init__(
        self,
        language,
        npy_arrays=None,
        root_dir="~/.cache/opensubtitles",
        overwrite=False,
        preprocess_function=None,
        first_n_lines=float("inf"),
        update_bar_every=10_000,
        n_sents=1,
        alt_len=True,
    ):
        self.n_sents = n_sents
        self.alt_len = alt_len
        if npy_arrays is not None:
            self.slices = npy_arrays[0]
            self.lengths = npy_arrays[1]
        else:
            slices_file = root_dir + "slices.npy"
            lengths_file = root_dir + "lengths.npy"
            slices = deque()
            offset = 0
            self.update_bar_every = update_bar_every
            self.slices_file = slices_file
            self.lenghts_file = lengths_file
            txt_file = f"{root_dir}/{language}/{language}.txt"
            if path.isfile(txt_file + ".pre") and not overwrite:
                txt_file = txt_file + ".pre"
            elif preprocess_function and (not path.isfile(slices_file) or overwrite):
                with open(txt_file, "rb") as f_read:
                    with open(txt_file + ".pre", "wb") as f_write:
                        for i, text in enumerate(
                            tqdm(
                                f_read,
                                desc="preprocessing file",
                                miniters=update_bar_every,
                                mininterval=5,
                            )
                        ):
                            if i >= first_n_lines - 1:
                                break
                            pre_text = preprocess_function(text.decode("utf-8"))
                            if pre_text is not None:
                                pre_text = pre_text.encode("utf-8")
                                f_write.write(pre_text)
                                slices.append(offset)
                                offset += len(pre_text)
                        slices.append(offset)
                txt_file = txt_file + ".pre"
                preprocessed = True
                self._convert_deque_to_npy(slices)
            if not path.isfile(slices_file) and not overwrite and ".pre" in txt_file:
                raise Exception(
                    "preprocessed file exists, but slices and lenghts do not",
                    "try preprocessing again",
                )
            if path.isfile(slices_file) and not overwrite:
                self.slices = np.load(slices_file)
                self.lengths = np.load(lengths_file)
                if first_n_lines * self.n_sents < len(self.slices):
                    self.slices = self.slices[: first_n_lines * self.n_sents]
                    self.lengths = self.lengths[: first_n_lines * self.n_sents]
                self._reshape_npy()
            elif not preprocessed:
                with open(txt_file, "rb") as f:
                    for i, text in enumerate(
                        tqdm(
                            f,
                            desc="reading from file",
                            miniters=update_bar_every,
                            mininterval=5,
                        )
                    ):
                        if i >= first_n_lines - 1:
                            break
                        slices.append(offset)
                        offset += len(text)
                    slices.append(offset)
                self._convert_deque_to_npy(slices)
        with open(txt_file, "r+b") as f:
            if ".pre" in txt_file:
                self.trim_linebreak = 0
            else:
                self.trim_linebreak = 1
            self.mm = mmap.mmap(f.fileno(), 0)

    def _reshape_npy(self):
        if self.n_sents > 1:
            max_i = len(self.slices) // self.n_sents * self.n_sents
            self.slices = np.reshape(self.slices[:max_i], (-1, self.n_sents))[:, 0]
            self.lengths = np.reshape(self.lengths[:max_i], (-1, self.n_sents))

    @staticmethod
    def _insert_blank(mystring, position):
        mystring = mystring[:position] + b" " + mystring[position:]
        return mystring

    def _convert_deque_to_npy(self, slices):
        self.slices = np.empty(len(slices) - 1, dtype="uint64")
        self.lengths = np.empty(len(slices) - 1, dtype="ushort")
        i = 1
        with tqdm(total=len(slices), desc="converting to numpy array") as pbar:
            self.slices[0] = slices.popleft()
            pbar.update(1)
            while len(slices) > 1:
                next_val = slices.popleft()
                self.lengths[i - 1] = next_val - self.slices[i - 1]
                self.slices[i] = next_val
                i += 1
                if i % self.update_bar_every == 0:
                    pbar.update(self.update_bar_every)
            last_val = slices.popleft()
            self.lengths[i - 1] = last_val - self.slices[i - 1]
            del slices
            pbar.update(i % self.update_bar_every)
            pbar.close()
        np.save(self.lenghts_file, self.lengths)
        np.save(self.slices_file, self.slices)
        self._reshape_npy()

    def __len__(self):
        if self.alt_len and self.n_sents > 1:
            return len(self.slices) * 2
        else:
            return len(self.slices)

    def __getitem__(self, idx):
        orig_idx = idx
        idx = idx // 2
        start = self.slices[idx]
        end = start + self.lengths[idx].sum() - self.trim_linebreak
        result = self.mm[int(start) : int(end)]
        shift_arr = np.arange(self.n_sents - 1)
        if self.n_sents > 1 and not self.alt_len:
            for l in self.lengths[idx].cumsum()[:-1] + shift_arr:
                result = OpenSubtitlesDataset._insert_blank(result, int(l))
        elif self.alt_len:
            mod = idx % (self.n_sents - 1)
            offset_arr = self.lengths[idx].cumsum()[:-1]
            result1 = self.mm[int(start) : int(start + offset_arr[mod])]
            length1 = self.lengths[idx][: mod + 1].sum()
            result2 = self.mm[int(start + offset_arr[mod]) : int(end)]
            length2 = self.lengths[idx][mod + 1 :].sum()
            for i, l in enumerate(offset_arr):
                if i < mod:
                    result1 = OpenSubtitlesDataset._insert_blank(
                        result1, int(l + shift_arr[i])
                    )
                elif i > mod:
                    result2 = OpenSubtitlesDataset._insert_blank(
                        result2, int(l + shift_arr[i - mod - 1] - offset_arr[mod])
                    )
            if orig_idx % 2 == 0:
                return result1.decode("utf-8"), length1
            else:
                return result2.decode("utf-8"), length2
        return result.decode("utf-8"), self.lengths[idx].sum()

    def _get_alt_lens(self):
        self_len = len(self.lengths)
        odd_tiles = np.tile(
            np.tril(np.ones((self.n_sents, self.n_sents)), -1)[:, :-1],
            self_len // (self.n_sents - 1) + 1,
        ).T[:self_len]
        even_tiles = np.tile(
            np.triu(np.ones((self.n_sents, self.n_sents)), 1)[:, 1:],
            self_len // (self.n_sents - 1) + 1,
        ).T[:self_len]
        even_lens = (even_tiles * self.lengths).sum(axis=1)
        odd_lens = (odd_tiles * self.lengths).sum(axis=1)
        return np.ravel(np.column_stack((even_lens, odd_lens)))

    def splits(self,  split=[0.7, 0.15, 0.15], seed=None, sort_by_len=True):
        if seed is not None:
            torch.manual_seed(seed)
        lens = [int(len(self) * s) for s in split]
        missing = len(self) - np.sum(lens)
        lens[np.argmax(split)] += int(missing)
        rand_splits = []
        if not sort_by_len:
            return random_split(self, lens)
        for split in random_split(self, lens):
            ind = np.array(split.indices)
            if self.alt_len and self.n_sents > 1:
                lengths = self._get_alt_lens()[ind]
            else:
                lengths = self.lengths[ind].sum(axis=1)
            lengthsort = lengths.argsort()
            split.indices = list(ind[lengthsort])
            rand_splits.append(split)
        return rand_splits
