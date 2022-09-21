from typing import List, Tuple

LABEL_NAMES = [
    'TL', 'TR', 'BL', 'BR', 'Shoulder', 'Clavicle', 'Pelvis', 'Iliac',
    'FemoralHead'
]
LABEL_NAMES += ['C7-TL', 'C7-TR', 'S-TL', 'S-TR']

FLIPPED_NAMES = [
    'TR', 'TL', 'BR', 'BL', 'Shoulder', 'Clavicle', 'Pelvis', 'Iliac',
    'FemoralHead'
]
FLIPPED_NAMES += ['C7-TR', 'C7-TL', 'S-TR', 'S-TL']

flip_pairs = []
for i, (ln, fn) in enumerate(zip(LABEL_NAMES, FLIPPED_NAMES)):
    if ln != fn and (i not in set([e[1] for e in flip_pairs])):
        flip_pairs.append((i, FLIPPED_NAMES.index(ln)))
FLIP_PAIRS = flip_pairs