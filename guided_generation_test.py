import os
import re
from itertools import chain
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataclasses import dataclass
from typing import List, Optional, Callable
from tqdm.auto import tqdm

from lambdag.lambdag import LambdaG


@dataclass
class BeamHypothesis:
    """Container for a single beam hypothesis"""

    tokens: List[int]
    score: float
    log_probs: List[float]
    lambdag_score: float  # Single score for the entire sequence
    is_finished: bool = False


class GuidedBeamSearch:
    """Beam search with custom guidance."""

    def __init__(
        self,
        model,
        tokenizer,
        lambdag_score: Callable,
        beam_size: int = 4,
        max_length: int = 500,
        alpha: float = 0.5,  # Weight for combining scores
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
        device="cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lambdag_score = lambdag_score
        self.beam_size = beam_size
        self.max_length = max_length
        self.alpha = alpha  # Weight: alpha * likelihood + (1-alpha) * lambdag
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.eos_token_id = eos_token_id or tokenizer.eos_token_id
        self.device = device
        self.input_length = 0  # Will be set during generation

    def _get_initial_beams(self, input_ids: torch.Tensor) -> List[BeamHypothesis]:
        """Initialize beams with the input sequence"""
        initial_tokens = input_ids[0].tolist()
        initial_text = self.tokenizer.decode(initial_tokens, skip_special_tokens=True)
        return [
            BeamHypothesis(
                tokens=initial_tokens, score=0.0, log_probs=[], lambdag_score=0
            )
        ]

    def _expand_beam(
        self, beam: BeamHypothesis, logits: torch.Tensor, step: int
    ) -> List[BeamHypothesis]:
        """Expand a single beam with top-k next tokens"""
        if self.temperature != 1.0:
            logits = logits / self.temperature
        log_probs = F.log_softmax(logits, dim=-1)
        top_k = min(self.beam_size * 2, log_probs.size(-1))  # Consider more candidates
        top_log_probs, top_indices = torch.topk(log_probs, top_k)

        candidates = []
        for log_prob, token_id in zip(top_log_probs.tolist(), top_indices.tolist()):
            # Create new hypothesis
            new_tokens = beam.tokens + [token_id]
            new_log_probs = beam.log_probs + [log_prob]

            # Calculate average log likelihood
            avg_log_likelihood = sum(new_log_probs) / len(new_log_probs)

            # Get lambdag score for the new sequence
            token_sequence = self.tokenizer.decode(
                # We ignore the input when calculating lambdaG
                new_tokens[self.input_length :],
                skip_special_tokens=True,
            )
            lambdag_val = self.lambdag_score(token_sequence)

            # Combine scores with length penalty
            # length_factor = ((5 + len(new_tokens)) / 6) ** self.length_penalty
            length_factor = len(new_tokens) ** self.length_penalty
            combined_score = (
                self.alpha * avg_log_likelihood + (1 - self.alpha) * lambdag_val
            ) * length_factor

            new_beam = BeamHypothesis(
                tokens=new_tokens,
                score=combined_score,
                log_probs=new_log_probs,
                lambdag_score=lambdag_val,
                is_finished=(token_id == self.eos_token_id),
            )

            candidates.append(new_beam)

        return candidates

    def generate(self, input_text: str, num_return_sequences: int = 1) -> List[str]:
        """Generate text using guided beam search"""
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
            self.device
        )
        self.input_length = input_ids.shape[1]
        active_beams = self._get_initial_beams(input_ids)
        finished_beams = []
        for step in tqdm(range(self.max_length)):
            # Get all candidates from all beams
            all_candidates = []

            for beam in tqdm(active_beams, leave=False):
                if beam.is_finished:
                    finished_beams.append(beam)
                    continue

                # Prepare input
                beam_input = torch.tensor([beam.tokens], device=self.device)

                # Get model output
                with torch.no_grad():
                    outputs = self.model(beam_input)
                    next_token_logits = outputs.logits[0, -1, :]

                # Expand beam
                candidates = self._expand_beam(beam, next_token_logits, step)
                all_candidates.extend(candidates)

            # Select top beams
            if all_candidates:
                all_candidates.sort(key=lambda x: x.score, reverse=True)
                active_beams = all_candidates[: self.beam_size]
            else:
                break

            # Check if all beams are finished
            if all(beam.is_finished for beam in active_beams):
                finished_beams.extend(active_beams)
                break

        # Add remaining active beams to finished
        finished_beams.extend([b for b in active_beams if not b.is_finished])

        # Sort by score and return top sequences
        finished_beams.sort(key=lambda x: x.score, reverse=True)

        results = []
        for i in range(min(num_return_sequences, len(finished_beams))):
            text = self.tokenizer.decode(
                finished_beams[i].tokens, skip_special_tokens=True
            )
            results.append(text)

        return results


def extract_paragraphs(text):
    """
    Extract paragraphs from Gutenberg plain-text format.

    Args:
        text (str): The input text in Gutenberg format

    Returns:
        list: A list of paragraphs, with each paragraph as a string with no line breaks
    """
    # Split the text by double newlines (paragraph separator in Gutenberg texts)
    raw_paragraphs = text.split("\n\n")

    # Process each paragraph to join lines within paragraphs
    paragraphs = []
    for para in raw_paragraphs:
        # Use regex to replace any sequences of whitespace (including newlines) with a single space
        clean_para = re.sub(r"\s+", " ", para).strip()
        # Only add non-empty paragraphs
        if clean_para and len(clean_para.split()) > 20:
            paragraphs.append(clean_para.replace("_", ""))

    return paragraphs


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['BNB_CUDA_VERSION'] = "120"
    
    parser = argparse.ArgumentParser(
        description="Run guided beam search with optional 8-bit quantization"
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization with BitsAndBytesConfig",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Beam size for beam search (default: 4)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=500,
        help="Maximum number of new tokens to generate (default: 500)",
    )

    args = parser.parse_args()

    texts = []
    for root, _, files in os.walk("data"):
        for f in files:
            path = os.path.join(root, f)
            if "Dickens" in path and "checkpoint" not in path:
                with open(path, "r", encoding="utf-8") as inp:
                    texts.append(extract_paragraphs(inp.read()))

    model_name = "ibm-granite/granite-3.3-2b-instruct"  # NB: chat template
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir="../hf_cache/", token=os.environ["HF_TOKEN_GATED"]
    )
    test_input_user = [
        {
            "role": "user",
            "content": "Write an obituary for Abraham Lincoln in the style of Charles Dickens. Only output the obituary itself without a title or a byline.",
        }
    ]
    test_input = tokenizer.apply_chat_template(
        test_input_user, add_generation_prompt=False, tokenize=False
    )

    # Configure quantization if requested
    quantization_config = None
    if args.use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Using 8-bit quantization")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="../hf_cache/",
        token=os.environ["HF_TOKEN_GATED"],
        quantization_config=quantization_config,
        device_map="auto",
    )
    inputs = tokenizer(test_input, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, num_beams=args.beam_size, max_new_tokens=args.max_new_tokens
    )
    print("Vanilla-beam-search output:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    lambda_g = LambdaG(
        disable_multiprocessing=True,  # For quicker online processing of short texts
        disable_tqdm=True,
    )
    lambda_g.train_known_author_model(list(chain.from_iterable(texts)), "Dickens")

    gbs = GuidedBeamSearch(
        model,
        tokenizer,
        lambda s: lambda_g.compute_lambda_g(s, "Dickens"),
        beam_size=args.beam_size,
        max_length=args.max_new_tokens,
        device="cuda",
    )
    print("Guided-beam-search output:")
    print(gbs.generate(test_input))


if __name__ == "__main__":
    main()
