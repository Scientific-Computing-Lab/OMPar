import argparse
import json
import torch
from compAI import OMPAR


if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

    parser.add_argument('--vocab_file', default='tokenizer/gpt/gpt_vocab/gpt2-vocab.json')
    parser.add_argument('--merge_file', default='tokenizer/gpt/gpt_vocab/gpt2-merges.txt')
    parser.add_argument('--model_weights', help='Path to the OMPify model weights', required=True)

    main_args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ompar = OMPAR(model_path=main_args.model_weights, device=device, args=main_args)


    with open('use_cases.jsonl', 'r') as f:
        for line in f:
            sample = json.loads(line.strip())

            code = sample['code']
            gt_pragma = sample['pragma']
            label = sample['label']

            pred_pragma = ompar.auto_comp(code)
            
            print(f'Code:\n{code}')
            print(f'GT pragma: {gt_pragma}')
            print(f"Pred pragma: {'' if not pred_pragma else '#pragma '+pred_pragma}")
            print('-'*20)
