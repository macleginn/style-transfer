import unittest
import json
import os
from train_pos_tagger_on_xglue import detokenize_list
import string_constants

class TestDetokenizeList(unittest.TestCase):
    def test_empty_list(self):
        """Test that empty list of tokens returns empty string."""
        self.assertEqual(detokenize_list([]), "")
    
    def test_single_token(self):
        """Test that a single token is handled correctly."""
        self.assertEqual(detokenize_list(["Hello"]), "Hello")
    
    def test_basic_punctuation(self):
        """Test basic punctuation handling - no spaces before punctuation."""
        self.assertEqual(detokenize_list(["Hello", "."]), "Hello.")
        self.assertEqual(detokenize_list(["Hello", ",", "world", "!"]), "Hello, world!")
        self.assertEqual(detokenize_list(["One", ":", "two", ";", "three"]), "One: two; three")
    
    def test_quotes(self):
        """Test handling of quotes with proper spacing rules."""
        # Opening quotes should have space before (unless after opening bracket)
        self.assertEqual(detokenize_list(["He", "said", "\"", "hello", "\"", "."]), 
                         "He said \"hello\".")
        
        # No space inside quotes
        self.assertEqual(detokenize_list(["\"", "Hello", "world", "\"", "."]), 
                         "\"Hello world\".")
        
        # Opening brackets followed by quotes don't need space
        self.assertEqual(detokenize_list(["(", "\"", "Hello", "\"", ")"]), 
                         "(\"Hello\")")
    
    def test_apostrophes(self):
        """Test handling of apostrophes in contractions."""
        # Test apostrophe at the end
        self.assertEqual(detokenize_list(["The", "boy", "'"]), "The boy'")
        
        # Test apostrophe at beginning followed by a word
        self.assertEqual(detokenize_list(["'", "tis"]), "'tis")
        
        # Test possessives and contractions
        self.assertEqual(detokenize_list(["Today", "'s", "news"]), "Today's news")
        self.assertEqual(detokenize_list(["I", "did", "n't", "go"]), "I didn't go")
    
    def test_hyphens(self):
        """Test handling of hyphens in compound words."""
        self.assertEqual(detokenize_list(["wheel", "-", "chair"]), "wheel-chair")
        self.assertEqual(detokenize_list(["F", "-", "16", "-", "launched"]), "F-16-launched")
        self.assertEqual(detokenize_list(["pre", "-", "trained", "model"]), "pre-trained model")
    
    def test_opening_symbols(self):
        """Test handling of opening symbols that should not be followed by spaces."""
        self.assertEqual(detokenize_list(["Look", "(", "here", ")"]), "Look (here)")
        self.assertEqual(detokenize_list(["Items", ":", "[", "one", ",", "two", "]"]), 
                         "Items: [one, two]")
    
    def test_mixed_punctuation(self):
        """Test mixed punctuation scenarios."""
        self.assertEqual(
            detokenize_list(["She", "asked", ",", "\"", "Why", "?", "\"", "."]),
            "She asked, \"Why?\"."
        )
        self.assertEqual(
            detokenize_list(["He", "(", "the", "older", "one", ")", "left", "."]),
            "He (the older one) left."
        )
    
    def test_dev_examples(self):
        self.maxDiff = None
        """Test with examples from dev.json file."""
        # Load examples from dev.json
        dev_file_path = os.path.join(os.path.dirname(__file__), 'pos_data', 'dev.json')
        with open(dev_file_path, 'r') as f:
            examples = [json.loads(line) for line in f.readlines()]
        
        # Test a selection of examples
        test_cases = [
            # Basic example with simple punctuation
            examples[0]["tokens"],  # ["From","the","AP","comes","this","story",":"]
            
            # Example with Google analyst day
            examples[16]["tokens"],  # Google has finally had an analyst day...
            
            # Example with hyphens
            examples[5]["tokens"],  # Contains "wheel-chair"
            
            # Example with apostrophes (contractions)
            examples[6]["tokens"],  # Contains "n't" and other contractions
            
            # Example with "Today's" possessive
            examples[7]["tokens"],  # Contains "Today's"
            
            # Example with parentheses
            examples[13]["tokens"],  # Contains parentheses
            
            # Example with commas and more complex structure
            examples[1]["tokens"],  # Contains commas, periods, and multiple clauses
            
            # Example with quotes
            examples[18]["tokens"],  # Contains quotes
            
            # Example with colons
            examples[4]["tokens"],  # Contains colons and more punctuation
        ]
        
        # Expected results (precomputed for clarity)
        expected_results = [
            "From the AP comes this story:",
            "Google has finally had an analyst day -- a chance to present the company's story to the (miniscule number of) people who haven't heard it.",
            "The sheikh in wheel-chair has been attacked with a F-16-launched bomb.",
            "He could be killed years ago and the israelians have all the reasons, since he founded and he is the spiritual leader of Hamas, but they didn't.",
            "Today's incident proves that Sharon has lost his patience and his hope in peace.",
            "(I hope that the US army got an enormous amount of information from her relatives, because otherwise this move was a bad, bad tradeoff).",
            "President Bush on Tuesday nominated two individuals to replace retiring jurists on federal courts in the Washington area.",
            "They work on Wall Street, after all, so when they hear a company who's stated goals include \"Don't be evil,\" they imagine a company who's eventually history will be \"Don't be profitable.\"",
            "Bush also nominated A. Noel Anketell Kramer for a 15-year term as associate judge of the District of Columbia Court of Appeals, replacing John Montague Steadman."
        ]
        
        for i, tokens in enumerate(test_cases):
            detokenized = detokenize_list(tokens)
            self.assertEqual(detokenized, expected_results[i], 
                            f"Failed on example {i}: {tokens}")
    
    def test_all_dev_examples(self):
        """Test all examples from dev.json to ensure no exceptions are raised."""
        dev_file_path = os.path.join(os.path.dirname(__file__), 'pos_data', 'dev.json')
        with open(dev_file_path, 'r') as f:
            examples = [json.loads(line) for line in f.readlines()]
        
        for example in examples:
            # Just ensure no exceptions are raised
            detokenized = detokenize_list(example["tokens"])
            self.assertIsInstance(detokenized, str)
    
    def test_edge_cases(self):
        """Test edge cases and unusual punctuation patterns."""
        # Multiple consecutive punctuation marks
        self.assertEqual(detokenize_list(["Wait", "...", "what", "?", "!"]), 
                         "Wait... what?!")
        
        # Multiple quotes --- unsupported
        # self.assertEqual(
        #     detokenize_list(["He", "said", "\"", "She", "said", "\"", "Hello", "\"", "\"", "."]),
        #     "He said \"She said \"Hello\"\".") 
        
        # Nested quotes with different types --- unsupported for '-marked internal quotes
        self.assertEqual(
            detokenize_list(["\"", "He", "whispered", "'", "Hello", "'", "\"", "."]),
            "\"He whispered ' Hello '\".")

if __name__ == '__main__':
    unittest.main()