import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json


def get_loader(transform,
               mode='val',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               num_workers=0,
               cocoapi_loc='/opt',
               generate_gt=False):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'val' or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary. 
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading 
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """

    assert mode in ['val', 'test'], "mode must be one of 'val' or 'test'."
    assert batch_size==1, "Please change batch_size to 1 if testing your model."
    assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."

    if mode == 'test':
        img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/test2014/')
        annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/image_info_test2014.json')
    else:
        img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/val2014/')
        annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_val2014.json')

    # COCO caption dataset.
    dataset = CoCoDataset(transform=transform,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          img_folder=img_folder,
                          generate_gt=generate_gt)

    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=dataset.batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    return data_loader


class CoCoDataset(data.Dataset):
    
    def __init__(self, transform, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, img_folder, generate_gt):
        self.transform = transform
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, True)
        self.img_folder = img_folder
        self.generate_gt = generate_gt
        self.coco = COCO(annotations_file)
        self.ids = list(self.coco.anns.keys())
        
    def __getitem__(self, index):
        ann_id = self.ids[index]  
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']
        
        # Convert image to tensor and pre-process using transform
        pil_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        orig_image = np.array(pil_image)
        image = self.transform(pil_image)

        annotations = []
        if self.generate_gt:
            for ann in self.coco.imgToAnns[img_id]:
                annotations.append(ann['caption'])
            # print(annotations)
            
        return orig_image, image, img_id, annotations

    def __len__(self):
        return len(self.ids)