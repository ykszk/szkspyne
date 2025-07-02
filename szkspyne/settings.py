import json
import os
from typing import Any


def load_env_var(name: str, default: Any) -> Any:
    '''
    Load environmental variable stored as json string
    '''
    var = os.environ.get(name, default)
    if isinstance(var, str):
        return json.loads(var)
    else:
        return var


VERTEBRA_LABEL_NAMES = load_env_var('VERTEBRA_LABEL_NAMES', ['TL', 'TR', 'BL', 'BR'])
ASM_LABEL_NAMES = load_env_var('ASM_LABEL_NAMES', ['TL', 'TR', 'BL', 'BR'])
NON_VERTEBRA_LABEL_NAMES = load_env_var(
    'NON_VERTEBRA_LABEL_NAMES', ['Shoulder', 'Clavicle', 'Pelvis', 'Iliac', 'FemoralHead']
)
ANCHOR_LABEL_NAMES = load_env_var('ANCHOR_LABEL_NAMES', ['C7-TL', 'C7-TR', 'S-TL', 'S-TR'])
IMPLANT_LABEL_NAMES = load_env_var('IMPLANT_LABEL_NAMES', ['Rod', 'Screw', 'Hook', 'Transverse'])
LABEL_NAMES = load_env_var('LABEL_NAMES', VERTEBRA_LABEL_NAMES + NON_VERTEBRA_LABEL_NAMES + ANCHOR_LABEL_NAMES)
# max number of points for each label. TL and TR have max 18 points (T1 to T12, L1 to L6, and S1)
MAX_NUM_POINTS = load_env_var(
    'MAX_NUM_POINTS',
    '{"TL":19, "TR":19, "BL":18, "BR":18, "Shoulder": 2, "Clavicle": 2, "Pelvis": 2, "Iliac": 2, "FemoralHead": 2, "C7-TL": 1, "C7-TR": 1, "S-TL": 1, "S-TR": 1}',
)

FLIPPED_NAMES = load_env_var(
    'FLIPPED_NAMES', ['TR', 'TL', 'BR', 'BL'] + NON_VERTEBRA_LABEL_NAMES + ['C7-TR', 'C7-TL', 'S-TR', 'S-TL']
)


PROB_MAX = 4000


USE_NECK_SETTINGS = load_env_var('USE_NECK_SETTINGS', False)

def set_neck_settings():
    global LABEL_NAMES, FLIPPED_NAMES, ANCHOR_LABEL_NAMES, ASM_LABEL_NAMES, MAX_NUM_POINTS
    LABEL_NAMES = ["TL", "TR", "BL", "BR", "Brow", "Sella", "ExternalAuditoryCanal", "Orbit", "Occipital", "Chin", "Lamina", "Manubrium", "PosteriorHardPalate", "AnteriorC1Arch", "AnteriorDens", "PosteriorDens", "Lamina1", "C3_BL", "C3_BR"]
    FLIPPED_NAMES = ["TR", "TL", "BR", "BL", "Brow", "Sella", "ExternalAuditoryCanal", "Orbit", "Occipital", "Chin", "Lamina", "Manubrium", "PosteriorHardPalate", "AnteriorC1Arch", "AnteriorDens", "PosteriorDens", "Lamina1", "C3_BR", "C3_BL"]
    ANCHOR_LABEL_NAMES = ["AnteriorDens", "PosteriorDens", "Lamina1", "C3_BL", "C3_BR"]
    ASM_LABEL_NAMES = ["TL", "TR", "BL", "BR", "Lamina", "AnteriorC1Arch", "AnteriorDens", "PosteriorDens"]
    MAX_NUM_POINTS = { "TL": 7, "TR": 7, "BL": 8, "BR": 8, "Brow": 1, "Sella": 1, "ExternalAuditoryCanal": 1, "Orbit": 1, "Occipital": 1, "Chin": 1, "Lamina": 8, "Manubrium": 1, "PosteriorHardPalate": 1, "AnteriorC1Arch": 1, "AnteriorDens": 1, "PosteriorDens": 1, "C3_BL": 1, "C3_BR": 1, "Lamina1": 1 }

if USE_NECK_SETTINGS:
    set_neck_settings()


def create_flip_pairs(normal, flipped):
    flip_pairs = []
    for i, (ln, fn) in enumerate(zip(normal, flipped)):
        if ln != fn and (i not in set([e[1] for e in flip_pairs])):
            flip_pairs.append((i, flipped.index(ln)))
    return flip_pairs


if set(LABEL_NAMES) > set(FLIPPED_NAMES):
    import sys

    print('Invalid pair of LABEL_NAMES and FLIPPED_NAMES.')
    print(LABEL_NAMES)
    print(FLIPPED_NAMES)
    sys.exit(1)

FLIP_PAIRS = create_flip_pairs(LABEL_NAMES, FLIPPED_NAMES)
