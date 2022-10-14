VERTEBRA_LABEL_NAMES = ['TL', 'TR', 'BL', 'BR']
NON_VERTEBRA_LABEL_NAMES = [
    'Shoulder', 'Clavicle', 'Pelvis', 'Iliac', 'FemoralHead'
]
ANCHOR_LABEL_NAMES = ['C7-TL', 'C7-TR', 'S-TL', 'S-TR']
LABEL_NAMES = VERTEBRA_LABEL_NAMES + NON_VERTEBRA_LABEL_NAMES + ANCHOR_LABEL_NAMES

FLIPPED_NAMES = ['TR', 'TL', 'BR', 'BL'
                 ] + NON_VERTEBRA_LABEL_NAMES + ANCHOR_LABEL_NAMES

flip_pairs = []
for i, (ln, fn) in enumerate(zip(LABEL_NAMES, FLIPPED_NAMES)):
    if ln != fn and (i not in set([e[1] for e in flip_pairs])):
        flip_pairs.append((i, FLIPPED_NAMES.index(ln)))
FLIP_PAIRS = flip_pairs