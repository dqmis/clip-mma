from scripts.run_experiment import run_experiment

NUM_OF_SAMPLES = 16

TRAIN_SUBSAMPLES = ["base"]

MODEL_TYPES = [
    "clip_mha_adapter",
    "clip_transformer_adapter",
    "clip_mha_adapter_mlp_bottleneck",
    "clip_transformer_adapter_mlp_bottleneck",
]


DATASETS = [
    "cifar10",
    "oxford_flowers",
    "oxford_pets",
    "sun397",
    "dtd",
    "caltech101",
    "fgvc_aircraft",
    "food101",
    "stanford_cars",
]

run_experiment(datasets=DATASETS, model_types=MODEL_TYPES, train_subsamples=TRAIN_SUBSAMPLES)
