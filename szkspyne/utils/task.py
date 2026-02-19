


CMD_SORT_VERTEBRAL_CORNERS = ['lmrs', 'sort', '-', '--labels', 'TL,TR,BL,BR']
'''Sort vertebral corner by y coordinate'''

CMD_SORT_NON_VERTEBRAL = CMD_SORT_VERTEBRAL_CORNERS + ['--inv-label', '--by-x']
'''Sort non-vertebral points by x coordinate'''