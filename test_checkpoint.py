import torch
from torch.utils.data import DataLoader, Dataset
from fomo.utils.data.datasets import DatasetInitializer
from fomo.pipelines.utils.initializers import (
    initalize_dataloaders,
    initalize_datasets,
    intialize_model,
)

from fomo.pipelines.train import Learner
from fomo.pipelines.types.learner_args import LearnerArgs


if __name__ == "__main__":
    dataset_name = 'cifar10'
    
    
    learner_args = LearnerArgs()
    learner_args.model_type = "clip_extension"
    learner_args.train_size = 0.8

    learner = Learner(learner_args)


    checkpoint = torch.load('/home/schatterjee/FOMO/fomo/output/clip_base_vit-b16_1715101582/checkpoint.pth.tar')
    learner.model.load_state_dict(checkpoint['state_dict'])

    
    learner.evaluate(split='test')