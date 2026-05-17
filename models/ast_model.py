import torch
import torch.nn as nn

from transformers import ASTForAudioClassification


class ASTModel(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=4,
            ignore_mismatched_sizes=True
        )

        # Freeze all
        for param in self.model.parameters():
            param.requires_grad = False

        # 🔥 ONLY LAST TRANSFORMER BLOCK
        for name, param in self.model.named_parameters():

            if (
                "encoder.layer.11" in name
                or "classifier" in name
            ):
                param.requires_grad = True

        self.dropout = nn.Dropout(0.4)

    def forward(self, x):

        outputs = self.model(x).logits

        outputs = self.dropout(outputs)

        return outputs