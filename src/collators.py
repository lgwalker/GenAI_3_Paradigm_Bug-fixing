import torch
import random

class ManualSpanCorruptionCollator:
    """Implements T5 span corruption: groups 15% of masked tokens into spans."""
    def __init__(self, tokenizer, corruption_rate=0.15):
        self.tokenizer = tokenizer
        self.corruption_rate = corruption_rate
        self.sentinel_ids = tokenizer.convert_tokens_to_ids(
            [f"<extra_id_{i}>" for i in range(100)]
        )

    def __call__(self, batch):
        input_ids_batch, labels_batch = [], []

        for item in batch:
            token_ids = item["input_ids"]
            n_tokens = len(token_ids)
            n_mask = max(1, int(n_tokens * self.corruption_rate))
            mask_indices = sorted(random.sample(range(n_tokens), n_mask))

            spans = []
            if mask_indices:
                current_span = [mask_indices[0]]
                for i in range(1, len(mask_indices)):
                    if mask_indices[i] == mask_indices[i - 1] + 1:
                        current_span.append(mask_indices[i])
                    else:
                        spans.append(current_span)
                        current_span = [mask_indices[i]]
                spans.append(current_span)

            mask_set = set(mask_indices)
            span_map = {idx: s_idx for s_idx, span in enumerate(spans) for idx in span}

            input_ids, prev_span_idx = [], -1
            for pos, tid in enumerate(token_ids):
                if pos in mask_set:
                    s_idx = span_map[pos]
                    if s_idx != prev_span_idx:
                        input_ids.append(self.sentinel_ids[s_idx])
                        prev_span_idx = s_idx
                else:
                    input_ids.append(tid)
                    prev_span_idx = -1

            labels = []
            for s_idx, span in enumerate(spans):
                labels.append(self.sentinel_ids[s_idx])
                for pos in span:
                    labels.append(token_ids[pos])
            labels.append(self.tokenizer.eos_token_id)

            input_ids_batch.append(torch.tensor(input_ids))
            labels_batch.append(torch.tensor(labels))

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=0)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True, padding_value=-100)

        print("Span corruption collator initialized  |  Corruption rate: 15%")
        
        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": (input_ids_padded != 0).long(),
        }
