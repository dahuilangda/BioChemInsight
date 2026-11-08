import os
import json
import random
import numpy as np
import re
from SmilesPE.pretokenizer import atomwise_tokenizer

PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
MASK = '<mask>'
#UNK = '<unk>'
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
MASK_ID = 3


def is_atom_token(token):
    return token.isalpha() or token.startswith("[") or token == '*'

def split_keep_special(text, special_tokens=["<sep>", "<pad>"]):
    pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + r"|\S)"
    return re.findall(pattern, text)

class CharTokenizer:

    def __init__(self, vocab_path, debug=False):
        super().__init__()

        with open(vocab_path) as f:
            self.stoi = json.load(f)
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

        self.debug = debug

    def __len__(self):
        return len(self.stoi)

    def smiles_to_sequence(self, smiles, coords=None, mask_ratio=0, atom_only=False, charwise_tokenizer=False):
        tokens = smiles if charwise_tokenizer else atomwise_tokenizer(smiles)
        if ''.join(tokens) != smiles:
            raise ValueError('Tokenizer did not preserve the complete input')
        sequence = [SOS_ID]
        atom_indices = []
        for token in tokens:
            if atom_only and not is_atom_token(token):
                continue
            for c in token:
                if c in self.stoi:
                    if mask_ratio > 0 and random.random() < mask_ratio:
                        sequence.append(MASK_ID)
                    else:
                        sequence.append(self.stoi[c])
                else:
                    raise ValueError(f'Character absent from vocabulary: {c!r}')
            if not charwise_tokenizer and is_atom_token(token):
                atom_indices.append(len(sequence) - 1)
        sequence.append(EOS_ID)
        return sequence, atom_indices

    def sequence_to_smiles(self, sequence: list) -> dict:
        # TODO: handle sequences with <sep> and ext
        smiles = ''
        atoms, indices = [], []
        i = 0
        while i < len(sequence):
            label = sequence[i]
            if label == EOS_ID or label == PAD_ID:
                break
            if not is_atom_token(self.itos[label]):
                smiles += self.itos[label]
                i += 1
                continue
            
            # now handle atom tokens
            if self.itos[label] == '[':
                j = i + 1
                while j < len(sequence):
                    if sequence[j] == EOS_ID or sequence[j] == PAD_ID:
                        break
                    if self.itos[sequence[j]] == ']':
                        j += 1
                        break
                    j += 1
            else:
                # handle Cl and Br which are two characters
                if i+1 < len(sequence) and (self.itos[label] == 'C' and self.itos[sequence[i+1]] == 'l' \
                        or self.itos[label] == 'B' and self.itos[sequence[i+1]] == 'r'):
                    j = i+2
                else:
                    j = i+1
            atom = ''.join(self.itos[sequence[k]] for k in range(i, j))
            smiles += atom

            if j <= len(sequence):
                atoms.append(atom)
                indices.append(j - 1)   # store the index of the last atom character
            i = j
        results = {'smiles': smiles, 'atoms': atoms, 'indices': indices}
        
        return results
