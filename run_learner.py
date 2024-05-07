from fomo.pipelines.train import Learner
from fomo.pipelines.types.learner_args import LearnerArgs



if __name__ == "__main__":

    learner_args = LearnerArgs()
    #learner_args.device = "cuda"
    #learner_args.epochs = 2
    learner_args.model_type = "clip_extension"
    learner_args.train_size = 0.8

    learner = Learner(learner_args)

    learner.run()