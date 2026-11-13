import json
import random
import re

from SmilesPE.pretokenizer import atomwise_tokenizer

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
MASK = "<mask>"
# UNK = '<unk>'
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


def is_atom_token(token):
    return token.isalpha() or token.startswith("[") or token == "*"


def split_keep_special(text, special_tokens=None):
    if special_tokens is None:
        special_tokens = ["<sep>", "<pad>"]
    pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + r"|\S)"
    return re.findall(pattern, text)


class CharTokenizer:
    def __init__(self, vocab_path, debug=False):
        super().__init__()

        with open(vocab_path) as f:
            self.stoi = json.load(f)
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        if sorted(self.itos) != list(range(len(self.stoi))):
            raise ValueError("Vocabulary IDs must be unique and contiguous")
        if any(
            self.stoi.get(token) != index
            for token, index in ((PAD, PAD_ID), (SOS, SOS_ID), (EOS, EOS_ID))
        ):
            raise ValueError("Vocabulary has incompatible termination token IDs")
        self.mask_id = self.stoi.get(MASK)

        self.debug = debug

    def __len__(self):
        return len(self.stoi)

    def smiles_to_sequence(
        self,
        smiles,
        coords=None,
        mask_ratio=0,
        atom_only=False,
        charwise_tokenizer=False,
    ):
        base, separator, extension = (
            smiles.partition("<sep>") if not charwise_tokenizer else (smiles, "", "")
        )
        if not charwise_tokenizer and not separator and " |" in smiles:
            base, _, extension = smiles.partition(" |")
            separator, extension = " ", "|" + extension
        tokens = list(base) if charwise_tokenizer else atomwise_tokenizer(base)
        if "".join(tokens) != base:
            raise ValueError("Tokenizer did not preserve the complete input")
        if mask_ratio and self.mask_id is None:
            raise ValueError("Masking requires a mask token in the vocabulary")
        sequence = [SOS_ID]
        atom_indices = []
        for token in tokens:
            if atom_only and not is_atom_token(token):
                continue
            for c in token:
                if c in self.stoi:
                    if mask_ratio > 0 and random.random() < mask_ratio:
                        sequence.append(self.mask_id)
                    else:
                        sequence.append(self.stoi[c])
                else:
                    raise ValueError(f"Character absent from vocabulary: {c!r}")
            if not charwise_tokenizer and is_atom_token(token):
                atom_indices.append(len(sequence) - 1)
        if separator:
            if atom_only:
                raise ValueError("Atom-only encoding cannot preserve extended labels")
            if separator not in self.stoi:
                raise ValueError(
                    "Extended labels require their separator in the vocabulary"
                )
            sequence.append(self.stoi[separator])
            for character in extension:
                if character not in self.stoi:
                    raise ValueError(f"Character absent from vocabulary: {character!r}")
                sequence.append(self.stoi[character])
        sequence.append(EOS_ID)
        return sequence, atom_indices

    def sequence_to_smiles(self, sequence: list) -> dict:
        smiles = ""
        atoms, indices = [], []
        i = 0
        in_extension = False
        while i < len(sequence):
            label = sequence[i]
            if label == EOS_ID:
                break
            if label == PAD_ID:
                # Deliberate: a sampled PAD is an invalid terminal action.
                # Keeping "<pad>" in the decoded string fails parse and
                # zeroes the reward, so GRPO pushes probability away from
                # emitting it mid-sequence (pinned by test_tokenizer).
                smiles += PAD
                break
            if self.itos[label] in ("<sep>", " "):
                in_extension = True
            if in_extension:
                smiles += self.itos[label]
                i += 1
                continue
            if not is_atom_token(self.itos[label]):
                smiles += self.itos[label]
                i += 1
                continue

            # now handle atom tokens
            if self.itos[label] == "[":
                j = i + 1
                while j < len(sequence):
                    if sequence[j] == EOS_ID or sequence[j] == PAD_ID:
                        break
                    if self.itos[sequence[j]] == "]":
                        j += 1
                        break
                    j += 1
            else:
                # handle Cl and Br which are two characters
                if i + 1 < len(sequence) and (
                    self.itos[label] == "C"
                    and self.itos[sequence[i + 1]] == "l"
                    or self.itos[label] == "B"
                    and self.itos[sequence[i + 1]] == "r"
                ):
                    j = i + 2
                else:
                    j = i + 1
            atom = "".join(self.itos[sequence[k]] for k in range(i, j))
            smiles += atom

            if j <= len(sequence):
                atoms.append(atom)
                indices.append(j - 1)  # store the index of the last atom character
            i = j
        base, _, extension = smiles.partition("<sep>")
        if " |" in base:
            base, _, extension = smiles.partition(" |")
            extension = "|" + extension
        results = {
            "smiles": smiles,
            "atoms": atoms,
            "indices": indices,
            "base_smiles": base,
            "extension": extension,
        }

        return results
