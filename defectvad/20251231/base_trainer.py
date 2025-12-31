# common/base_trainer.py

from abc import ABC, abstractmethod
from tqdm import tqdm
import torch

from .early_stopper import EarlyStopper
from .evaluator import Evaluator


class BaseTrainer(ABC):
    def __init__(self, model, loss_fn=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, torch.nn.Module) else loss_fn

        self.train_loader = None
        self.valid_loader = None

        # configure optimizers
        self.optimizer = None
        self.scheduler = None
        self.gradient_clip_val = None

        # configure early stoppers
        self.train_early_stopper = None
        self.valid_early_stopper = None

        # training epochs and steps
        self.global_epoch = 0
        self.global_step = 0
        self.max_epochs = 1
        self.max_steps = 1

        self.evaluator = Evaluator(self.model, pixel_level=False)

    #######################################################
    # setup for anomaly detection models
    #######################################################

    def configure_optimizers(self):
        pass

    def configure_early_stoppers(self):
        self.train_early_stopper = None
        self.valid_early_stopper = EarlyStopper(patience=5, min_delta=1e-3, monitor="auroc")

    @abstractmethod
    def training_step(self, batch):
        raise NotImplementedError
   
    #######################################################
    # fit: train model for max_epochs or max_steps
    #######################################################

    def fit(self, train_loader, max_epochs=1, valid_loader=None):
        self.max_epochs = max_epochs
        self.max_steps = max_epochs * len(train_loader)
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.configure_optimizers()
        self.configure_early_stoppers()

        self.on_fit_start()
        self.on_train_start()

        for _ in range(self.max_epochs):
            self.on_train_epoch_start()
            train_outputs = self.train_one_epoch()
            self.on_train_epoch_end(train_outputs)

            if self.valid_loader is not None and self.evaluator is not None:
                self.on_validation_epoch_start()
                valid_outputs = self.validate_one_epoch()
                self.on_validation_epoch_end(valid_outputs)

            if self.train_early_stop or self.valid_early_stop:
                break

        self.on_train_end()
        self.on_fit_end()

    #######################################################
    # Hooks
    #######################################################

    def backward(self, loss):
        loss.backward()

    def on_fit_start(self): pass

    def on_train_start(self):
        self.early_stop_str = ""
        self.train_early_stop = False
        self.valid_early_stop = False
        self.current_epoch = 0
        self.current_step = 0
        print("\n*** Training start...")

    def on_train_epoch_start(self):
        self.global_epoch += 1
        self.current_epoch += 1

    def on_train_batch_start(self, batch, batch_idx): pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.global_step += 1
        self.current_step += 1

    def on_train_epoch_end(self, outputs):
        self.epoch_info = f"[{self.current_epoch:3d}/{self.max_epochs}]"
        self.train_info = ", ".join([f"{k}:{v:.3f}" for k, v in outputs.items()])
        if self.valid_loader is None:
            print(f"{self.epoch_info} {self.train_info}")

        if self.train_early_stopper is not None:
            metric_name = self.train_early_stopper.monitor
            self.train_early_stop = self.train_early_stopper.step(outputs[metric_name])

            if self.train_early_stopper.target_reached:
                self.early_stop_str += f"Training target readched! {self.train_early_stopper.get_info()}"
            elif self.train_early_stopper.early_stop:
                self.early_stop_str += f"Training Early Stopped! {self.train_early_stopper.get_info()}"

        # if self.max_steps is not None:
        #     if self.current_step >= self.max_steps:
        #         self.train_early_stop = True
        #         self.early_stop_str += f"Max training step reached! {self.current_step} steps"

    def on_validation_epoch_start(self): pass

    def on_validation_batch_start(self, batch, batch_idx): pass

    def on_validation_batch_end(self, outputs, batch, batch_idx): pass

    def on_validation_epoch_end(self, outputs):
        valid_info = ", ".join([f"{k}:{v:.3f}" for k, v in outputs.items()])
        print(f"{self.epoch_info} {self.train_info} | (val) {valid_info}")

        if self.valid_early_stopper is not None:
            metric_name = self.valid_early_stopper.monitor
            self.valid_early_stop = self.valid_early_stopper.step(outputs[metric_name])

            if self.valid_early_stopper.target_reached:
                self.early_stop_str += f"Validation target readched! {self.valid_early_stopper.get_info()}"
            elif self.valid_early_stopper.early_stop:
                self.early_stop_str += f"Validation Early Stopped! {self.valid_early_stopper.get_info()}"

    def on_train_end(self):
        if self.train_early_stop or self.valid_early_stop:
            print(f">> {self.early_stop_str}")
        print("*** Training completed!")

    def on_fit_end(self): pass

    #######################################################
    # Train one epoch
    #######################################################

    @torch.enable_grad()
    def train_one_epoch(self):
        self.model.train()
        outputs = {}
        num_images = 0

        with tqdm(self.train_loader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(f">> Training")
            for batch_idx, batch in enumerate(progress_bar):
                self.on_train_batch_start(batch, batch_idx)

                batch_size = batch["image"].shape[0]
                num_images += batch_size
                batch_outputs = self.training_step(batch)

                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    loss = batch_outputs["loss"]
                    self.backward(loss)

                    if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
                        torch.nn.utilsclip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_val)

                    self.optimizer.step()

                    if self.scheduler is not None:
                        self.scheduler.step()

                for name, value in batch_outputs.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    outputs.setdefault(name, 0.0)
                    outputs[name] += value * batch_size

                progress_bar.set_postfix({name: f"{value / num_images:.3f}" for name, value in outputs.items()})
                self.on_train_batch_end(batch_outputs, batch, batch_idx)

        return {name: value / num_images for name, value in outputs.items()}

    #######################################################
    # Validate one epoch
    #######################################################

    @torch.no_grad()
    def validate_one_epoch(self):
        return self.evaluator(self.valid_loader) if self.evaluator else {}
        
