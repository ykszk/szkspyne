# env vars in json
export LABEL_NAMES='["TL", "TR", "BL", "BR", "Brow", "Sella", "ExternalAuditoryCanal", "Orbit", "Occipital", "Chin", "Lamina", "Manubrium", "PosteriorHardPalate", "AnteriorC1Arch", "AnteriorDens", "PosteriorDens"]'
export FLIPPED_NAMES='["TR", "TL", "BR", "BL", "Brow", "Sella", "ExternalAuditoryCanal", "Orbit", "Occipital", "Chin", "Lamina", "Manubrium", "PosteriorHardPalate", "AnteriorC1Arch", "AnteriorDens", "PosteriorDens"]'
# BL and BR have max 7 (C2 to C7 and T1) points
export MAX_NUM_POINTS='{ "TL": 7, "TR": 7, "BL": 8, "BR": 8, "Brow": 1, "Sella": 1, "ExternalAuditoryCanal": 1, "Orbit": 1, "Occipital": 1, "Chin": 1, "Lamina": 1, "Manubrium": 1, "PosteriorHardPalate": 1, "AnteriorC1Arch": 1, "AnteriorDens": 1, "PosteriorDens": 1 }'