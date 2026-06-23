import torch
from typing import Any, Callable

from moviad.trainers.trainer import Trainer
from moviad.models.training_args import TrainingArgs
from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.evaluation.metrics import Metric
from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.models.filo.filo import FiLoModel, FiLoArgs


class FiLoTrainer(Trainer):
    """
    Trainer class for FiLo model.

    Implements the two-phase training strategy described in the paper:
    - Phase 1 (main): trains prompt learners and decoder for `epochs` epochs
    - Phase 2 (adapter): trains the adapter for `adapter_epochs` epochs
    """

    def __init__(
        self,
        train_args: FiLoArgs,
        model: FiLoModel,
        train_dataset: VADDataset,
        eval_dataset: VADDataset | None,
        metrics: list[Metric],
        device: torch.device,
        logger: Any | None = None,
        logging_prefix: str = "",
        save_path: str | None = None,
        saving_criteria: Callable | None = None,
    ):
        """
        Initialize FiLo trainer.

        Args:
            train_args: FiLo training arguments (FiLoArgs)
            model: FiLoModel instance
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            metrics: List of metrics to evaluate
            device: Device to run training on
            logger: Logger for tracking metrics
            logging_prefix: Prefix for logged metrics
            save_path: Path to save model checkpoints
            saving_criteria: Function to determine when to save model
        """

        self.optimizer_main_part = train_args.optimizer_main_part
        self.optimizer_adapter = train_args.optimizer_adapter

        super().__init__(
            train_args=train_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            metrics=metrics,
            device=device,
            logger=logger,
            logging_prefix=logging_prefix,
            save_path=save_path,
            saving_criteria=saving_criteria,
        )

    def train(self):

        if self.logger:
            self.logger.config.update(self.train_args.__to_dict__())

        best_metrics = {metric.name: 0.0 for metric in self.metrics}

        # ------------------------------------------------------------------ #
        # PHASE 1: Main training (prompt learners + decoder)
        # ------------------------------------------------------------------ #
        if not self.train_args.train_adapter:
            print("=== Phase 1: Main Training (Prompt Learners + Decoder) ===")

            for epoch in range(self.train_args.epochs):

                print(f"EPOCH: {epoch}")

                avg_batch_loss = self.model.train_epoch(
                    epoch,
                    self.train_dataloader,
                    self.train_args
                )
                print(f"Average training loss: {avg_batch_loss:.4f}")

                if self.logger:
                    self.logger.log({
                        f"{self.logging_prefix}train/epoch": epoch,
                        f"{self.logging_prefix}train/main_loss": avg_batch_loss
                    })

                if (epoch + 1) % self.train_args.evaluation_epoch_interval == 0:

                    print("Evaluating model...")
                    self.model.eval()

                    results = Evaluator.evaluate(
                        self.model,
                        self.eval_dataloader,
                        self.metrics,
                        self.device
                    )

                    self.save_model(best_metrics, results)
                    best_metrics = Trainer.update_best_metrics(best_metrics, results)

                    print("Training performances:")
                    Trainer.print_metrics(results)

                    if self.logger is not None:
                        self.logger.log({
                            f"{self.logging_prefix}/test/{metric_name}": value
                            for metric_name, value in results.items()
                        })
        else:
            # ------------------------------------------------------------------ #
            # PHASE 2: Adapter training
            # ------------------------------------------------------------------ #
            print("=== Phase 2: Adapter Training ===")

            for epoch in range(self.train_args.epochs):

                print(f"ADAPTER EPOCH: {epoch}")

                avg_batch_loss = self.model.train_epoch(
                    epoch,
                    self.train_dataloader,
                    self.train_args
                )
                print(f"Average adapter loss: {avg_batch_loss:.4f}")

                if self.logger:
                    self.logger.log({
                        f"{self.logging_prefix}train/adapter_epoch": epoch,
                        f"{self.logging_prefix}train/adapter_loss": avg_batch_loss
                    })

        # ------------------------------------------------------------------ #
        # Final save
        # ------------------------------------------------------------------ #
        if self.saving_criteria is None and self.save_path is not None:
            print("Saving final model...")
            self.model.save(self.save_path)
            print(f"Model saved to {self.save_path}")