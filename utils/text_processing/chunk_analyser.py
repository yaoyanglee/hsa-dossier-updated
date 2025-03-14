import logging

from statistics import mean, median
from typing import List, Dict, NamedTuple
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title


# --- Configure logger ---
# Create a named logger
logger = logging.getLogger(__name__)

# Configure the logger
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class ChunkSettings(NamedTuple):
    max_characters: int
    combine_under_chars: int
    overlap: int

class ChunkAnalyser:
    def __init__(self, target_min_chunk: int = 2000, ideal_num_chunks: int = 8, 
                 min_chunk_weight: float = 0.3, num_chunks_weight: float = 0.4, consistency_weight: float = 0.3):
        self.target_min_chunk = target_min_chunk
        self.ideal_num_chunks = ideal_num_chunks
        self.weights = {
            'min_chunk': min_chunk_weight,
            'num_chunks': num_chunks_weight,
            'consistency': consistency_weight
        }

    def analyse_chunks(self, filename: str, max_chars_options: List[int]) -> ChunkSettings:
        elements = partition_pdf(filename=filename, strategy="hi_res")
        results = self._test_chunk_configs(elements, max_chars_options)
        best_config = self._get_best_config(results)
        self._log_analysis(results, best_config)
        return ChunkSettings(
            max_characters=best_config,
            combine_under_chars=best_config,
            overlap=best_config // 2
        )

    def _test_chunk_configs(self, elements: List, max_chars_options: List[int]) -> Dict:
        results = {}
        for max_chars in max_chars_options:
            chunks = chunk_by_title(
                elements, max_characters=max_chars, combine_text_under_n_chars=max_chars, overlap=max_chars // 2)
            chunk_lengths = [len(str(chunk)) for chunk in chunks]
            results[max_chars] = {
                'num_chunks': len(chunks),
                'avg_length': mean(chunk_lengths),
                'median_length': median(chunk_lengths),
                'max_length': max(chunk_lengths),
                'min_length': min(chunk_lengths),
                'length_ratio': min(chunk_lengths) / max(chunk_lengths)
            }
        return results

    def _score_config(self, stats: Dict) -> float:
        min_chunk_score = min(1.0, stats['min_length'] / self.target_min_chunk)
        num_chunks_score = 1.0 / (1.0 + abs(stats['num_chunks'] - self.ideal_num_chunks))
        consistency_score = stats['length_ratio']
        return (
            self.weights['min_chunk'] * min_chunk_score +
            self.weights['num_chunks'] * num_chunks_score +
            self.weights['consistency'] * consistency_score
        )

    def _get_best_config(self, results: Dict) -> int:
        scores = {max_chars: self._score_config(stats) for max_chars, stats in results.items()}
        return max(scores.items(), key=lambda x: x[1])[0]

    def _log_analysis(self, results: Dict, best_config: int) -> None:
        logger.info("=== Chunking Analysis ===")
        for max_chars, stats in results.items():
            logger.info(f"Config max_chars={max_chars}: {stats}")
        logger.info(f"Recommended configuration: max_characters={best_config}")
