"""generate paper submission tables and plots
for eccv
"""
from re import M
from collections import OrderedDict

def _get_path(key, dataset="nci"):
    """dataset: nci, brats"""
    return "{}/{}/train-rs42-rsTR1.log".format("/data/<USERNAME>/checkpoints/eccv/eccv_{}_v2".format(dataset), key)


EXP_KEY = "compare_sota"
# store the data file and experiment name key = experiment (including rs), val = {(rsTR, filepath)}

METHOD_MAPPING = OrderedDict()
# PLOT_LINESTYLES = [
#     "-.",
#     "--",
#     "--",
#     "--",
#     "--",
#     "--",
#     "--",
#     "-"


# ]

# PLOT_MARKERS = [
#     ".",
#     "o",
#     "v",
#     "^",
#     "<",
#     ">",
#     "s",
#     "P",
# ]


PLOT_LINESTYLES = [
    "-.",
    "-.",
    "-.",
    "-.",
    "-",
    "-",    
]

PLOT_MARKERS = [
    ".",
    "o",
    "v",
    "^",  
    "s",
    "P",
]


# nci Component Ablations
# METHOD_MAPPING.update({
#     "ft-rp4-nit150": "Fine Tuning",
#     "ft-rp4-nit150-kdH-kdL-lwf0-kdM1-kdX0.4-kdT1.0": "KD",
#     "ft-rp4-nit150-kdH-kdL-lwf0-kdM1-kdX0.4-kdT1.0-conL-conX0.1-conT1.0-conE0" : "KD+enc",
#     "ft-rp4-nit150-kdH-kdL-lwf0-kdM1-kdX0.4-kdT1.0-conL-conX0.1-conT1.0-conE0-coneL-coneX0.1-coneT1.0-coneE0": "KD+enc+neck",
#     "ft-rp4-nit150-conH-kdH-kdL-lwf0-kdM1-kdX0.4-kdT1.0-conL-conX0.1-conT1.0-conE0-coneL-coneX0.1-coneT1.0-coneE0-conbrL-conbrX0.1-coneT1.0-coneE0": "KD+enc+neck+neck2",
#     "ft-rp4-nit150-conH-kdH-kdL-lwf0-kdM1-kdX0.4-kdT1.0-conL-conX0.1-conT1.0-conE0-coneL-coneX0.1-coneT1.0-coneE0-conbrL-conbrX0.1-coneT1.0-coneE0-conlL-conlX0.1-conlT1.0-conlE0": "KD+enc+neck+neck2+dec",
        
# })

# nci One Hot Ablations
METHOD_MAPPING.update({
    "ft-rp4-nit150": "Fine Tuning",
    "ft-rp4-nit150-conH-kdH-kdL-lwf0-kdM0-kdX0.4-kdT1.0-conL-conX0.1-conT1.0-conE0-coneL-coneX0.1-coneT1.0-coneE0-conbrL-conbrX0.1-coneT1.0-coneE0-conlL-conlX0.1-conlT1.0-conlE0": "MSMT base",
    "ft-rp4-nit150-conH-kdH-kdL-lwf0-kdM1-kdX0.4-kdT1.0-conL-conX0.1-conT1.0-conE0-coneL-coneX0.1-coneT1.0-coneE0-conbrL-conbrX0.1-coneT1.0-coneE0-conlL-conlX0.1-conlT1.0-conlE0": "+KDM",
    "ft-rp4-nit150-conH-kdH-kdL-lwf0-kdM1-kdX0.4-kdT1.0-conL-conX0.1-conT1.0-conE0-coneL-coneX0.1-coneT1.0-coneE1-conbrL-conbrX0.1-coneT1.0-coneE0-conlL-conlX0.1-conlT1.0-conlE0": "+KDM+enc EF",
    "ft-rp4-nit150-conH-kdH-kdL-lwf0-kdM1-kdX0.4-kdT1.0-conL-conX0.1-conT1.0-conE0-coneL-coneX0.1-coneT1.0-coneE0-conbrL-conbrX0.1-coneT1.0-coneE0-conlL-conlX0.1-conlT1.0-conlE1": "+KDM+dec EF",
    "ft-rp4-nit150-conH-kdH-kdL-lwf0-kdM1-kdX0.4-kdT1.0-conL-conX0.1-conT1.0-conE0-coneL-coneX0.1-coneT1.0-coneE1-conbrL-conbrX0.1-coneT1.0-coneE0-conlL-conlX0.1-conlT1.0-conlE1": "+KDM+enc EF+dec EF",
        
})

PAPER_DATA = OrderedDict()
PAPER_DATA.update({
    k: (1, _get_path(k)) for k, _ in METHOD_MAPPING.items()
})









