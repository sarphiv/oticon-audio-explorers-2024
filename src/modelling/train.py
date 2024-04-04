from dataclasses import dataclass


@dataclass(frozen=True)
class ModelArgs:
    threshold: float = 0.5
    n_channels: int = 3
    n_classes: int = 2

    learning_rate_max: float = 1e-2
    learning_rate_min: float = 1e-3
    learning_rate_half_period: int = 2000
    learning_rate_mult_period: int = 2
    learning_rate_warmup_max: float = 4e-2
    learning_rate_warmup_steps: int = 1000
    weight_decay: float = 1e-6


@dataclass(frozen=True)
class Args:
    run_name: str | None = None
    batch_size: int = 128
    dataloader_workers: int = 8
    
    logging_step_period: int = 20
    checkpoint_save_n_best: int = 3
    checkpoint_save_every_n_steps: int = 4500

    model: ModelArgs = ModelArgs()



if __name__ == "__main__":
    # from pathlib import Path

    import tyro
    import torch as th
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger  # type: ignore

    from modelling.data.data_module import DataModule
    from modelling.model import Model


    # Set seeds
    pl.seed_everything(1337)


    # Load arguments
    args = tyro.cli(Args)

    # Set up logger
    logger = WandbLogger(
        project="oticon-2024", 
        entity="metrics_logger",
        name=args.run_name,
    )

    # Log arguments
    logger.experiment.config.update(args)


    # Set up data module
    dm = DataModule(
        validation_split=0.1,
        batch_size=args.batch_size,
        num_workers=args.dataloader_workers,
        shuffle=True,
        pin_memory=True
    )

    # Set up trainer
    th.set_float32_matmul_precision("medium")

    trainer = pl.Trainer(
        accelerator="auto",
        precision="16-mixed",
        logger=logger,
        log_every_n_steps=args.logging_step_period,
        callbacks=[
            LearningRateMonitor(
                logging_interval="step"
            ),
            ModelCheckpoint(
                dirpath="models/{logger.experiment.name}:{logger.experiment.hash}/",
                filename=f"{logger.experiment.id}" + ":top:{epoch:02d}:{step}:{val_loss:.3f}",
                every_n_train_steps=args.checkpoint_save_every_n_steps,
                save_top_k=args.checkpoint_save_n_best,
                mode="min",
                monitor="val_loss",
            ),
            ModelCheckpoint(
                dirpath="models",
                filename=f"{logger.experiment.id}" + ":all:{epoch:02d}:{step}:{val_loss:.3f}",
                every_n_train_steps=args.checkpoint_save_every_n_steps,
                save_top_k=-1,
            ),
        ]
    )


    # Set up model
    model = Model(
        threshold=args.model.threshold,
        n_channels=args.model.n_channels,
        n_classes=args.model.n_classes,
        learning_rate_max=args.model.learning_rate_max,
        learning_rate_min=args.model.learning_rate_min,
        learning_rate_half_period=args.model.learning_rate_half_period,
        learning_rate_mult_period=args.model.learning_rate_mult_period,
        learning_rate_warmup_max=args.model.learning_rate_warmup_max,
        learning_rate_warmup_steps=args.model.learning_rate_warmup_steps,
        weight_decay=args.model.weight_decay
    )

    # Start training, resume from checkpoint
    trainer.fit(model, dm)
