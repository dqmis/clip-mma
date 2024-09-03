from fomo.pipelines.train import Learner
from fomo.pipelines.types.learner_args import LearnerArgs
from fomo.utils.data.datasets import DatasetInitializer, PROMPTS

NUM_OF_SAMPLES = 16


def run_experiment(model_types, train_subsamples, datasets, prompts=None, num_samples=NUM_OF_SAMPLES):
    if prompts is None:
        prompts = PROMPTS
    for dataset_name in datasets:
        for model_type in model_types:
            for train_subsample in train_subsamples:
                try:
                    print(f"Training {model_type} on {dataset_name} on {train_subsample} subsample.")

                    _dataset = DatasetInitializer.from_str(dataset_name).value(train=True)
                    labels = _dataset.labels
                    prompt_template = prompts[dataset_name]
                    num_of_classes = len(labels) // 2 if train_subsample == "base" else len(labels)
                    # num_of_classes = len(labels) - (len(labels) // 5) if train_subsample == "base" else len(labels)
                    train_eval_size = [num_samples * num_of_classes, num_samples * num_of_classes]

                    learner_args = LearnerArgs(model_type=model_type, dataset=dataset_name)
                    learner_args.model_type = model_type
                    learner_args.train_eval_size = train_eval_size
                    learner_args.text_prompt_template = prompt_template
                    learner_args.train_subsample = train_subsample
                    learner_args.dataset = dataset_name

                    learner = Learner(learner_args)
                    learner.run()
                except Exception as e:
                    print(f"Error: failed running - {str(e)}")
