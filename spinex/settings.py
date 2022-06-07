from typing import List, Tuple

LABEL_NAMES = [
    'TL', 'TR', 'BL', 'BR', 'Shoulder', 'Clavicle', 'Pelvis', 'Iliac',
    'FemoralHead'
]
FLIPPED_NAMES = [
    'TR', 'TL', 'BR', 'BL', 'Shoulder', 'Clavicle', 'Pelvis', 'Iliac',
    'FemoralHead'
]


def get_flip_pairs() -> List[Tuple[int, int]]:
    flip_pairs = []
    for i, (ln, fn) in enumerate(zip(LABEL_NAMES, FLIPPED_NAMES)):
        if ln != fn and (i not in set([e[1] for e in flip_pairs])):
            flip_pairs.append((i, FLIPPED_NAMES.index(ln)))
    return flip_pairs
