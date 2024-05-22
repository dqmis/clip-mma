from fomo.pipelines.train import Learner
from fomo.pipelines.types.learner_args import LearnerArgs



if __name__ == "__main__":

    learner_args = LearnerArgs()
    #learner_args.device = "cuda"
    #learner_args.epochs = 2
    learner_args.model_type = "clip_transformer"

    #learner_args.train_eval_size = (8*102,2*102)
    
    learner_args.train_eval_size = (16*37,8*37)
    #learner_args.dataset = "OXFORD_FLOWERS"
    learner_args.dataset = "OXFORD_PETS"
    #learner_args.text_prompt_template =  'a photo of a {}, a type of flower.'
    #learner_args.text_prompt_template =  'a photo of a {}, a type of pet.'
    #learner_args.text_prompt_template = 'This flower is a {}.'

    learner = Learner(learner_args)

    learner.run()