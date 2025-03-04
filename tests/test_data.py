import unittest
from transformers import AutoTokenizer
from typing import Dict, List, Any
import numpy as np

# Import the functions to be tested
from ddmoe.data.preprocess import sft_olmoe_train_batch_preprocess_fn


class TestIntegrationSFTPreprocess(unittest.TestCase):
    def setUp(self):
        # Load the actual tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125", trust_remote_code=True)

        # Sample data for testing
        self.sample_data = {
            "messages": [
                [
                    {"role": "user", "content": "Is 123 a prime?"},
                    {"role": "assistant", "content": "No, 123 is not a prime number. It can be factored as 3 × 41."}
                ],
                [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital of France is Paris."}
                ]
            ]
        }

    def test_integration_preprocessing(self):
        """Test the entire preprocessing pipeline with real tokenizer"""
        result = sft_olmoe_train_batch_preprocess_fn(self.sample_data, self.tokenizer)

        # Check the basic structure of the result
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertIn('labels', result)
        self.assertEqual(len(result['input_ids']), 2)

        # For each example, validate the labels
        for i, (input_ids, labels) in enumerate(zip(result['input_ids'], result['labels'])):
            # Print the tokens for debugging
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            # Identify token positions
            assistant_pos = -1
            for j, token in enumerate(tokens):
                if token == "<im_assistant>":
                    assistant_pos = j
                    break

            self.assertGreater(assistant_pos, 0, "Could not find <im_assistant> token")

            # Find the <im_middle> that follows <im_assistant>
            middle_pos = -1
            for j in range(assistant_pos + 1, len(tokens)):
                if tokens[j] == "<im_middle>":
                    middle_pos = j
                    break

            self.assertGreater(middle_pos, assistant_pos, "Could not find <im_middle> after <im_assistant>")

            # Find the <im_end> that follows the middle_pos
            end_pos = -1
            for j in range(middle_pos + 1, len(tokens)):
                if tokens[j] == "<im_end>":
                    end_pos = j
                    break

            self.assertGreater(end_pos, middle_pos, "Could not find <im_end> after <im_middle>")

            # Verify that labels are set correctly
            for j in range(len(labels)):
                if middle_pos < j <= end_pos:
                    # Assistant response should have labels matching input_ids
                    self.assertEqual(labels[j], input_ids[j],
                                     f"Label at position {j} should be {input_ids[j]} but got {labels[j]}")
                else:
                    # Everything else should be -100
                    self.assertEqual(labels[j], -100,
                                     f"Label at position {j} should be -100 but got {labels[j]}")

            # Verify assistant response by decoding
            # Extract just the assistant response part by decoding the tokens from middle_pos+1 to end_pos
            response_tokens = input_ids[middle_pos + 1:end_pos]
            decoded_response = self.tokenizer.decode(response_tokens)
            expected_response = self.sample_data["messages"][i][1]["content"]

            # The decoded response might not be exactly the same as the original due to tokenization
            # but it should contain the core content
            self.assertIn(expected_response.split()[0], decoded_response,
                          f"Expected to find beginning of '{expected_response}' in decoded '{decoded_response}'")

    def test_empty_messages(self):
        """Test handling of empty message lists"""
        empty_data = {"messages": []}
        result = sft_olmoe_train_batch_preprocess_fn(empty_data, self.tokenizer)

        # Should return empty lists
        self.assertEqual(len(result['input_ids']), 0)
        self.assertEqual(len(result['attention_mask']), 0)
        self.assertEqual(len(result['labels']), 0)

    def test_missing_roles(self):
        """Test with messages missing required roles"""
        incomplete_data = {
            "messages": [
                [
                    {"role": "user", "content": "Question without assistant?"}
                ],
                [
                    {"role": "assistant", "content": "Answer without question."}
                ],
                [
                    {"role": "system", "content": "Just a system message"}
                ]
            ]
        }

        result = sft_olmoe_train_batch_preprocess_fn(incomplete_data, self.tokenizer)

        # Should handle gracefully by skipping invalid messages
        self.assertEqual(len(result['input_ids']), 0)

    def test_long_sequences(self):
        """Test with very long messages that might exceed model context"""
        long_text = "This is a very long message. " * 100
        long_data = {
            "messages": [
                [
                    {"role": "user", "content": long_text},
                    {"role": "assistant", "content": "Short answer to a long question."}
                ]
            ]
        }

        result = sft_olmoe_train_batch_preprocess_fn(long_data, self.tokenizer)

        # Should process without errors
        self.assertEqual(len(result['input_ids']), 1)

        # Check if truncation works as expected (if enabled in the function)
        if hasattr(self.tokenizer, 'model_max_length'):
            self.assertLessEqual(len(result['input_ids'][0]), self.tokenizer.model_max_length)


if __name__ == '__main__':
    unittest.main()
