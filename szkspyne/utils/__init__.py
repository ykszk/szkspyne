import zlib
import numpy as np
from typing import Callable

class CompDecompGenerator:
    """
    Generator for compressing and decompressing a sequence of numpy arrays.
    """
    def __init__(self, arr: np.ndarray, compressor: Callable, decompressor: Callable = None):
        self.decompressor = decompressor
        self.dtype = arr.dtype
        self.arr = [(compressor(s.tobytes()), s.shape) for s in arr]

    def __iter__(self):
        return self

    def __next__(self):
        if not self.arr:
            raise StopIteration
        arr, shape = self.arr.pop(0)
        raw = self.decompressor(arr)
        return np.frombuffer(raw, dtype=self.dtype).reshape(shape)

    def __len__(self):
        return len(self.arr)

    @staticmethod
    def builder(method: str = 'zstandard', level: int = 1):
        """
        Builder method to create a compressor/decompressor based on the specified method.
        """
        if method == 'zlib':
            def compressor(s):
                return zlib.compress(s, level=level)
            def decompressor(s):
                return zlib.decompress(s)
        elif method == 'deflate':
            import deflate
            def compressor(s):
                return deflate.gzip_compress(s, level)
            def decompressor(s):
                return deflate.gzip_decompress(s)
        elif method == 'zstandard' or method == 'zstd':
            import zstandard
            def compressor(s):
                return zstandard.ZstdCompressor(level=level).compress(s)
            def decompressor(s):
                return zstandard.ZstdDecompressor().decompress(s)
        else:
            raise ValueError(f"Unsupported compression method: {method}")
        def builder(arr: np.ndarray):
            return CompDecompGenerator(arr, compressor=compressor, decompressor=decompressor)
        return builder