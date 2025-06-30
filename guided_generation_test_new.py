import os
import re
from itertools import chain
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataclasses import dataclass
from typing import List, Optional
from tqdm.auto import tqdm

from lambdag.lambdag import LambdaG


@dataclass
class BeamHypothesis:
    """Container for a single beam hypothesis"""

    tokens: List[int]
    score: float
    log_probs: List[float]
    lambdag_score: float  # Single score for the entire sequence
    lambdag_sum: float = 0.0  # Running sum of lambda_g_step scores
    lambdag_length: int = 0  # Number of n-grams contributing to the sum
    is_finished: bool = False


class GuidedBeamSearch:
    """Beam search with custom guidance."""

    def __init__(
        self,
        model,
        tokenizer,
        lambdag_scorer,  # Now expects a LambdaG object
        author_id: str,  # Author ID for the lambda_g_step calls
        beam_size: int = 4,
        max_length: int = 500,
        alpha: float = 0.5,  # Weight for combining scores
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
        device="cpu",
        averaging: bool = True,  # Whether to average lambda_g scores
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lambdag_scorer = lambdag_scorer
        self.author_id = author_id
        self.beam_size = beam_size
        self.max_length = max_length
        self.alpha = alpha  # Weight: alpha * likelihood + (1-alpha) * lambdag
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.eos_token_id = eos_token_id or tokenizer.eos_token_id
        self.device = device
        self.averaging = averaging
        self.input_length = 0  # Will be set during generation

    def _get_initial_beams(self, input_ids: torch.Tensor) -> List[BeamHypothesis]:
        """Initialize beams with the input sequence"""
        initial_tokens = input_ids[0].tolist()
        # initial_text = self.tokenizer.decode(initial_tokens, skip_special_tokens=True)
        return [
            BeamHypothesis(
                tokens=initial_tokens, 
                score=0.0, 
                log_probs=[], 
                lambdag_score=0,
                lambdag_sum=0.0,
                lambdag_length=0
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

            # Get lambda_g score for the new n-gram using lambda_g_step
            new_lambdag_sum = beam.lambdag_sum
            new_lambdag_length = beam.lambdag_length
            
            # Extract the generated portion (excluding input)
            generated_tokens = new_tokens[self.input_length:]
            
            # Convert tokens to text and then to a list for lambda_g_step
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # Tokenize using the lambda_g tokenizer for consistency
            tokenized_text = self.lambdag_scorer._tokenize_sentences(generated_text)[0]
            
            # lambda_g_step can handle short n-grams (they will be padded)
            if len(tokenized_text) > 0:
                lambdag_step_score = self.lambdag_scorer.lambda_g_step(
                    tokenized_text, self.author_id, clipping=True
                )
                new_lambdag_sum += lambdag_step_score
                new_lambdag_length += 1

            # Calculate final lambda_g score
            if new_lambdag_length > 0:
                if self.averaging:
                    lambdag_val = new_lambdag_sum / new_lambdag_length
                else:
                    lambdag_val = new_lambdag_sum
            else:
                lambdag_val = 1.0

            # Combine scores with length penalty
            length_factor = len(new_tokens) ** self.length_penalty
            combined_score = (
                self.alpha * avg_log_likelihood + (1 - self.alpha) * lambdag_val
            ) * length_factor

            new_beam = BeamHypothesis(
                tokens=new_tokens,
                score=combined_score,
                log_probs=new_log_probs,
                lambdag_score=lambdag_val,
                lambdag_sum=new_lambdag_sum,
                lambdag_length=new_lambdag_length,
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
            # Filter out finished beams
            unfinished_beams = [beam for beam in active_beams if not beam.is_finished]
            finished_beams.extend([beam for beam in active_beams if beam.is_finished])
            
            if not unfinished_beams:
                break

            # Get all candidates from all beams
            all_candidates = []


            # Batch compute logits for all unfinished beams
            if unfinished_beams:
                # Prepare batch input - pad sequences to same length
                beam_inputs = []
                max_len = max(len(beam.tokens) for beam in unfinished_beams)
                
                for beam in unfinished_beams:
                    # Pad with pad_token_id (or eos_token_id if pad_token_id is None)
                    pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    padded_tokens = beam.tokens + [pad_token_id] * (max_len - len(beam.tokens))
                    beam_inputs.append(padded_tokens)
                
                batch_input = torch.tensor(beam_inputs, device=self.device)
                
                # Get model output for all beams at once
                with torch.no_grad():
                    outputs = self.model(batch_input)
                    # Get logits for the last non-padded token of each sequence
                    batch_logits = []
                    for i, beam in enumerate(unfinished_beams):
                        last_token_idx = len(beam.tokens) - 1
                        batch_logits.append(outputs.logits[i, last_token_idx, :])
                    batch_logits = torch.stack(batch_logits)

            # Expand each beam using the pre-computed logits
            for i, beam in enumerate(unfinished_beams):
                candidates = self._expand_beam(beam, batch_logits[i], step + 1)
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
    # os.environ['BNB_CUDA_VERSION'] = "120"

    MAX_NEW_TOKENS_DEFAULT = 1000
    
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
        default=MAX_NEW_TOKENS_DEFAULT,
        help=f"Maximum number of new tokens to generate (default: {MAX_NEW_TOKENS_DEFAULT})",
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
    print('\n' + '-' * 80 + '\n')

    lambda_g = LambdaG(
        disable_tqdm=True,
        apply_pos_noise=False
    )
    lambda_g.train_known_author_model(list(chain.from_iterable(texts)), "Dickens")

    # Now we disable multiprocessing in lambda_g for quick online processing
    lambda_g.disable_multiprocessing = True

    gbs = GuidedBeamSearch(
        model,
        tokenizer,
        lambda_g,  # Pass the LambdaG object directly
        "Dickens",  # Author ID
        beam_size=args.beam_size,
        max_length=args.max_new_tokens,
        device="cuda",
        averaging=True,  # Enable averaging for lambda_g scores
    )
    print("Guided-beam-search output:")
    print(gbs.generate(test_input))


if __name__ == "__main__":
    main()
