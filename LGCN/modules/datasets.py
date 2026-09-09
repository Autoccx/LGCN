
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from .report_fields import (
    get_current_image_path,
    get_current_view,
    get_prior_image_path,
    get_prior_view,
    get_target_report,
    get_indication_text,
    get_structured_prior_report_text,
)


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.target_report_field = getattr(args, "target_report_field", "auto")
        self.use_pr_in_eval = bool(getattr(args, "use_prior_report_in_eval", False))
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r', encoding='utf-8').read())
        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            report = get_target_report(self.examples[i], self.target_report_field)
            self.examples[i]['ids'] = tokenizer(report)[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)

    def _load_image(self, rel_path):
        return Image.open(os.path.join(self.image_dir, rel_path)).convert('RGB')


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = self._load_image(image_path[0])
        image_2 = self._load_image(image_path[1])
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        extra = {
            "has_prior": False,
            "current_view": "unk",
            "prior_view": "unk",
            "indication_text": get_indication_text(example),
            "prior_report_text": "[NHPR]",
        }
        return image_id, image, report_ids, report_masks, seq_length, extra


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example.get('id', str(idx))

        cur_path = get_current_image_path(example)
        cur_view = get_current_view(example)
        cur_img = self._load_image(cur_path)

        prior_path = get_prior_image_path(example)
        prior_view = get_prior_view(example)
        has_prior = bool(prior_path)
        if has_prior:
            prior_img = self._load_image(prior_path)
        else:
            prior_img = cur_img.copy()

        if self.transform is not None:
            cur_img = self.transform(cur_img)
            prior_img = self.transform(prior_img)

        # [2, C, H, W], index 0=current anchor, index 1=latest prior or current-copy fallback
        image = torch.stack((cur_img, prior_img), 0)

        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        extra = {
            "has_prior": has_prior,
            "current_view": cur_view,
            "prior_view": prior_view,
            "indication_text": get_indication_text(example),
            "prior_report_text": get_structured_prior_report_text(
                example, self.split, use_pr_in_eval=self.use_pr_in_eval
            ),
        }
        return image_id, image, report_ids, report_masks, seq_length, extra
