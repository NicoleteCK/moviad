"""
Trainer for AnomalyCLIP model adapted to MOVIAD framework.
"""

import torch
from typing import Any, Callable

from moviad.trainers.trainer import Trainer
from moviad.models.training_args import TrainingArgs
from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.evaluation.metrics import Metric
from moviad.utilities.evaluation.evaluator import Evaluator


class AnomalyCLIPTrainer(Trainer):
    """
    Trainer class for AnomalyCLIP model.
    
    This trainer extends the base MOVIAD Trainer with specific
    configurations for AnomalyCLIP's prompt learning approach.
    """
    
    def __init__(
        self,
        train_args: TrainingArgs,
        model,  # AnomalyCLIPModel
        train_dataset: VADDataset,
        eval_dataset: VADDataset | None,
        metrics: list[Metric],
        device: torch.device,
        logger: Any | None = None,
        logging_prefix: str = "",
        save_path: str | None = None,
        saving_criteria: Callable | None = None,
        learning_rate: float = 0.001,
        betas: tuple = (0.5, 0.999)
    ):
        """
        Initialize AnomalyCLIP trainer.
        
        Args:
            train_args: Training arguments
            model: AnomalyCLIPModel instance
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            metrics: List of metrics to evaluate
            device: Device to run training on
            logger: Logger for tracking metrics
            logging_prefix: Prefix for logged metrics
            save_path: Path to save model checkpoints
            saving_criteria: Function to determine when to save model
            learning_rate: Learning rate for optimizer
            betas: Beta parameters for Adam optimizer
        """
        # Initialize optimizer before calling parent constructor
        # Only optimize prompt learner parameters
        self.optimizer = torch.optim.Adam(
            list(model.prompt_learner.parameters()),
            lr=learning_rate,
            betas=betas
        )
        
        # Update training args with optimizer
        train_args.optimizer = self.optimizer
        
        # Call parent constructor
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
            saving_criteria=saving_criteria
        )
    
    def train(self):
        """
        Main training loop with AnomalyCLIP-specific configurations.
        """
        # Initialize training (sets up optimizer in training_args)
        # Note: optimizer is already set in __init__
        
        if self.logger:
            self.logger.config.update(self.train_args.__to_dict__())
        
        best_metrics = {metric.name: 0.0 for metric in self.metrics}
        
        for epoch in range(self.train_args.epochs):
            # Keep CLIP model frozen, only train prompt learner
            self.model.model.eval()
            self.model.prompt_learner.train()
            
            print(f"EPOCH: {epoch}")
            
            # Training epoch
            avg_batch_loss = self.model.train_epoch(
                epoch, 
                self.train_dataloader, 
                self.train_args
            )
            
            # Log training loss
            if self.logger:
                self.logger.log({
                    f"{self.logging_prefix}train/epoch": epoch,
                    f"{self.logging_prefix}train/train_loss": avg_batch_loss
                })
            
            # Evaluation
            if (epoch + 1) % self.train_args.evaluation_epoch_interval == 0:
                print("Evaluating model...")
                
                # Set to eval mode for evaluation
                self.model.model.eval()
                self.model.prompt_learner.eval()
                
                results = Evaluator.evaluate(
                    self.model, 
                    self.eval_dataloader, 
                    self.metrics, 
                    self.device
                )
                
                # Save model if criteria met
                self.save_model(best_metrics, results)
                
                # Update best metrics
                best_metrics = Trainer.update_best_metrics(best_metrics, results)
                
                print("Training performances:")
                Trainer.print_metrics(results)
                
                # Log evaluation metrics
                if self.logger is not None:
                    if self.logging_prefix is not None:
                        self.logger.log({
                            f"{self.logging_prefix}/eval/{metric_name}": value 
                            for metric_name, value in results.items()
                        })
        
        # Final save if no criteria specified
        if self.saving_criteria is None and self.save_path is not None:
            print("Saving final model...")
            self.model.save(self.save_path)
            print(f"Model saved to {self.save_path}")
