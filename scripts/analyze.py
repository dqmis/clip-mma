from fomo.pipelines.analyze import Analyzer, AnalyzerArgs

if __name__ == "__main__":
    analyzer_args = AnalyzerArgs()
    analyzer_args.use_wandb = True
    analyzer_args.dataset = "cifar10"

    analyzer_args.model_type = "clip_transformer_adapter_text"
    # analyzer_args.dataloder_type = 'n_class_k_shot'
    # analyzer_args.n = 5
    # analyzer_args.k = 16
    analyzer_args.train_size = None  # 0.8
    analyzer_args.train_eval_size = [10, 10]
    analyzer_args.batch_size = 1
    analyzer_args.device = "cpu"

    analyzer_args.model_checkpoint_path = "checkpoints/model_best.pth.tar"

    analyzer = Analyzer(analyzer_args)

    print(f"Running analysis for model: {analyzer_args.model_type}...")

    analyzer.run()
