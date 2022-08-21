"""generate paper submission tables and plots
for eccv
"""
from re import M
from collections import OrderedDict

def _get_path(key, dataset="brats"):
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


# nci
# METHOD_MAPPING.update({
#     "ft-rp0-nit150": "Fine Tuning",
    
#     # "ft-rp0-nit150-kdL-lwf1-kdM0-kdX0.4-kdT2.0": "KD1",
#     # "ft-rp0-nit150-kdH-kdL-lwf0-kdM1-kdX0.4-kdT1.0": "KD2",
    
#     # "ft-rp0-nit150-coneL-coneX0.2-coneT2.0-coneE0": "Cont enc",
#     # "ft-rp0-nit150-conL-conX0.1-conT0.5-conE0": "Cont neck",
#     # "ft-rp0-nit150-conH-conbrL-conbrX0.2-coneT2.0-coneE0": "Cont neck2",
#     # "ft-rp0-nit150-conlL-conlX0.4-conlT1.0-conlE0": "Cont dec",
    
#     # "ft-rp0-nit150-conH-kdH-kdL-lwf0-kdM1-kdX0.4-kdT1.0-conL-conX0.1-conT1.0-conE0-coneL-coneX0.1-coneT1.0-coneE0-conbrL-conbrX0.1-coneT1.0-coneE0-conlL-conlX0.1-conlT1.0-conlE0": "MSMT",
        
    
#     "ft-rp2-nit150": "Mem2",
#     "ft-rp4-nit150": "Mem4",
#     "ft-rp64-nit150": "Joint Training",
#     "ft-rp2-nit150-conH-kdH-kdL-lwf0-kdM1-kdX0.4-kdT1.0-conL-conX0.1-conT1.0-conE0-coneL-coneX0.1-coneT1.0-coneE1-conbrL-conbrX0.1-coneT1.0-coneE0-conlL-conlX0.1-conlT1.0-conlE1": "Mem2+MSMT",    
#     "ft-rp4-nit150-conH-kdH-kdL-lwf0-kdM1-kdX0.4-kdT1.0-conL-conX0.1-conT1.0-conE0-coneL-coneX0.1-coneT1.0-coneE1-conbrL-conbrX0.1-coneT1.0-coneE0-conlL-conlX0.1-conlT1.0-conlE0": "Mem4+MSMT",
    
# })



# brats
METHOD_MAPPING.update({
    "ft-rp0-nit150" : "Fine Tuning",
    # "ft-rp0-nit150-kdL-lwf1-kdM0-kdX0.4-kdT2.0": "KD1",
    # "ft-rp0-nit150-kdH-kdL-lwf0-kdM1-kdX0.4-kdT1.0": "KD2",
    
    # "ft-rp0-nit150-coneL-coneX0.1-coneT0.5-coneE0": "Cont enc",
    # "ft-rp0-nit150-conL-conX0.1-conT0.5-conE0": "Cont neck",
    # "ft-rp0-nit150-conH-conbrL-conbrX0.2-coneT2.0-coneE0": "Cont neck2",
    # "ft-rp0-nit150-conlL-conlX0.4-conlT1.0-conlE0": "Cont dec",

    "ft-rp0-nit150-kdL-lwf1-kdM1-kdX0.4-kdT1.0-conL-conX0.1-conT0.5-conE0-coneL-coneX0.1-coneT0.5-coneE0-conlL-conlX0.1-conlT0.5-conlE0": "MSMT",
        
    "ft-rp2-nit150": "Mem2",
    "ft-rp4-nit150": "Mem4",
    # "ft-rp117-nit150": "Joint Training",
    # "ft-rp2-nit150-kdL-lwf1-kdM1-kdX0.4-kdT1.0-conL-conX0.1-conT0.5-conE0-coneL-coneX0.1-coneT0.5-coneE0-conlL-conlX0.1-conlT0.5-conlE0": "Mem2+MSMT",    
    # "ft-rp4-nit150-kdL-lwf1-kdM1-kdX0.4-kdT1.0-conL-conX0.1-conT0.5-conE0-coneL-coneX0.1-coneT0.5-coneE0-conlL-conlX0.1-conlT0.5-conlE0": "Mem4+MSMT",
})

PAPER_DATA = OrderedDict()
PAPER_DATA.update({
    k: (1, _get_path(k)) for k, _ in METHOD_MAPPING.items()
})









