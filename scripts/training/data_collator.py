"""
TrOCR Data Collator for HuggingFace Transformers
Based on Microsoft's official TrOCR implementation
"""

import torch
from dataclasses import dataclass
from typing import Any
from transformers import TrOCRProcessor

@dataclass
class TrOCRDataCollator:
    """
    Data collator for TrOCR training with VisionEncoderDecoderModel

    Handles:
    - pixel_values: Images (stacked without padding)
    - labels: Text tokens (pads with -100 for ignored tokens)
    """

    processor: TrOCRProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate a batch of TrOCR samples

        Args:
            features: List with samples from SwedishHandwritingDataset
                    Every sample: {'pixel_values': tensor, 'labels': tensor}
        
        Returns:
            Dict with 'pixel_values' and 'labels' for VisionEncoderDecoderModel
        """
        # Extract pixel_values and labels from samples
        pixel_values = [feature['pixel_values'] for feature in features]
        labels = [feature['labels'] for feature in features]

        # Stack images - Already same size from processor
        batch_pixel_values = torch.stack(pixel_values)

        # Find longest text sequence for padding
        max_length = max(len(label) for label in labels)

        # Pad all labels to same length
        padded_labels = []
        for label in labels:
            # Calculate padding needed
            padding_length = max_length - len(label)

            if padding_length > 0:
                # Pad with -100
                padded_label = torch.cat([
                    label,
                    torch.full((padding_length,), -100, dtype=label.dtype)
                ])
            else:
                padded_label = label

            padded_labels.append(padded_label)

        # Stack all padded labels
        batch_labels = torch.stack(padded_labels)

        return {
            'pixel_values': batch_pixel_values,
            'labels': batch_labels
        }
