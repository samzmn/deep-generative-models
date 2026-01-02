import os
import time
import csv
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any, List

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from visualize import plot_multiple_images
from utils import save_checkpoint, normal_noise_sampler, default_device


@dataclass
class EarlyStoppingConfig:
    monitor: str = "g_loss"          # metric name to monitor g_loss or d_loss
    patience: int = 10               # epochs to wait without improvement
    mode: str = "min"                # 'min' or 'max' (depending on metric)
    min_delta: float = 0.0           # minimum change to qualify as improvement
    start_from_epoch: int = 0        # ignore early stopping before this epoch


@dataclass
class TrainerConfig:
    out_dir: str
    epochs: int = 100
    device: torch.device = default_device()
    use_tensorboard: bool = True
    log_every_n_steps: Optional[int] = 100
    plot_every_n_epochs: Optional[int] = 1
    save_best_only: bool = True
    clip_grad_norm: Optional[float] = None
    seed: Optional[int] = None       # set for reproducibility


class CSVLogger:
    def __init__(self, csv_path: str, fieldnames: List[str]):
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_row(self, row: Dict[str, Any]):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class GANTrainer:
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        model_name: str,
        codings_dim: int,
        criterion: Callable[..., torch.Tensor],
        g_optimizer: Optimizer,
        d_optimizer: Optimizer,
        train_dataloader: DataLoader,
        dataset_name: str,
        noise_sampler: Callable[[int, int], torch.Tensor] | Any = normal_noise_sampler,  # returns latent z (and optionally cond)
        g_scheduler: Optional[LRScheduler] = None,   # torch.optim.lr_scheduler.LRScheduler
        d_scheduler: Optional[LRScheduler] = None,
        trainer_cfg: TrainerConfig = TrainerConfig(epochs=50, device=torch.device("cuda"), out_dir="./runs"),
        early_stopping: Optional[EarlyStoppingConfig] = EarlyStoppingConfig(),
        num_classes: Optional[int] = None, # must be set for conditional GANs
    ):
        self.G = generator.to(trainer_cfg.device)
        self.D = discriminator.to(trainer_cfg.device)
        self.model_name = model_name
        self.codings_dim = codings_dim
        self.criterion = criterion
        self.g_opt = g_optimizer
        self.d_opt = d_optimizer
        self.g_sch = g_scheduler
        self.d_sch = d_scheduler
        self.train_dl = train_dataloader
        self.dataset_name = dataset_name
        self.noise_sampler = noise_sampler
        self.cfg = trainer_cfg
        self.es = early_stopping
        self.num_classes = num_classes

        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
            torch.cuda.manual_seed_all(self.cfg.seed)

        os.makedirs(self.cfg.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cfg.out_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.cfg.out_dir, "images", f"{self.model_name}_{self.dataset_name}"), exist_ok=True)
        os.makedirs(os.path.join(self.cfg.out_dir, "ckpts"), exist_ok=True)
        os.makedirs(os.path.join(self.cfg.out_dir, "runs", f"{self.model_name}_{self.dataset_name}"), exist_ok=True)
        self.ckpt_path = None
        self.writer = None
        if self.cfg.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(self.cfg.out_dir, "runs", f"{self.model_name}_{self.dataset_name}"))

        # Tracking
        self.history = {
            "epoch": [],
            "g_loss": [],
            "d_loss": [],
            "epoch_sec": [],
            "g_lr": [],
            "d_lr": [],
        }

        self.best_metric = None
        self.best_epoch = -1
        self.no_improve_epochs = 0

        # CSV logger
        self.csv_logger = CSVLogger(
            os.path.join(self.cfg.out_dir, "logs", f"{self.model_name}_{self.dataset_name}_train_log.csv"),
            fieldnames=["epoch", "g_loss", "d_loss", "epoch_sec", "g_lr", "d_lr"]
        )

    def _is_better(self, current: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        if self.es.mode == "min":
            return (best - current) > self.es.min_delta
        else:  # "max"
            return (current - best) > self.es.min_delta

    def _get_lrs(self) -> Tuple[float, float]:
        g_lr = self.g_opt.param_groups[0]["lr"]
        d_lr = self.d_opt.param_groups[0]["lr"]
        return g_lr, d_lr

    def _save_checkpoint(self, epoch: int):
        if self.cfg.save_best_only:
            self.ckpt_path = os.path.join(self.cfg.out_dir, "ckpts", 
                                 f"{self.model_name}_{self.dataset_name}_best.pt")
        else:
            self.ckpt_path = os.path.join(self.cfg.out_dir, "ckpts", 
                                    f"{self.model_name}_{self.dataset_name}_epoch_{epoch}.pt")
        save_checkpoint(self.ckpt_path, epoch, self.G, self.D,
                        self.g_opt, self.d_opt)

    def _plot_curves(self):
        # Loss curves
        plt.figure(figsize=(7, 5))
        plt.plot(self.history["epoch"], self.history["g_loss"], label="G loss")
        plt.plot(self.history["epoch"], self.history["d_loss"], label="D loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training losses")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.out_dir, "runs", 
                                 f"{self.model_name}_{self.dataset_name}_loss_curves.png"))
        plt.close()

    def train(self):
        device = self.cfg.device
        fixed_z = self.noise_sampler(64, self.codings_dim).to(device)
        fixed_labels = torch.randint(0, self.num_classes, (64,), device=device) if self.num_classes is not None else None

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_start = time.time()
            self.G.train()
            self.D.train()

            g_running = 0.0
            d_running = 0.0
            step_count = 0

            for step, batch in enumerate(self.train_dl, start=1):
                # Support (images,) or (images, labels)
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    real_imgs, labels = batch[0].to(device), batch[1].to(device)
                    batch_size = real_imgs.size(0)
                else:
                    real_imgs = batch.to(device)
                    labels = None
                    batch_size = real_imgs.size(0)

                # Train Discriminator
                self.d_opt.zero_grad(set_to_none=True)
                codings = self.noise_sampler(batch_size, self.codings_dim).to(device)
                fake_imgs = self.G(codings).detach() if labels is None else self.G(codings, labels).detach()  # stop gradients to G for D step

                pred_real = self.D(real_imgs) if labels is None else self.D(real_imgs, labels)
                pred_fake = self.D(fake_imgs) if labels is None else self.D(fake_imgs, labels)
                # ones = torch.ones(batch_size, 1, device=device)
                # zeros = torch.zeros(batch_size, 1, device=device)
                ones = torch.empty(batch_size, 1, device=device).uniform_(0.9, 1.0)
                zeros = torch.empty(batch_size, 1, device=device).uniform_(0.0, 0.1)

                d_loss = self.criterion(pred_real, ones) + self.criterion(pred_fake, zeros)
                d_loss.backward()

                if self.cfg.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.cfg.clip_grad_norm)

                self.d_opt.step()

                # Train Generator
                self.g_opt.zero_grad(set_to_none=True)
                codings = self.noise_sampler(batch_size, self.codings_dim).to(device)
                gen_imgs = self.G(codings) if labels is None else self.G(codings, labels)
                for p in self.D.parameters():
                    p.requires_grad = False
                d_pred = self.D(gen_imgs) if labels is None else self.D(gen_imgs, labels)
                g_loss = self.criterion(d_pred, ones)

                g_loss.backward()

                if self.cfg.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.cfg.clip_grad_norm)

                self.g_opt.step()
                for p in self.D.parameters():
                    p.requires_grad = True
                g_running += float(g_loss.item())
                d_running += float(d_loss.item())
                step_count += 1

                if self.cfg.log_every_n_steps and step % self.cfg.log_every_n_steps == 0:
                    g_lr, d_lr = self._get_lrs()
                    print(f"\r[Epoch {epoch} | Step {step}] G_loss={g_loss.item():.4f} D_loss={d_loss.item():.4f} | G_lr={g_lr:.2e} D_lr={d_lr:.2e}", end="")

            # End of epoch: scheduler steps
            if self.g_sch is not None:
                self.g_sch.step()
            if self.d_sch is not None:
                self.d_sch.step()

            # Aggregate epoch stats
            g_epoch_loss = g_running / max(step_count, 1)
            d_epoch_loss = d_running / max(step_count, 1)
            epoch_sec = time.time() - epoch_start
            g_lr, d_lr = self._get_lrs()

            self.history["epoch"].append(epoch)
            self.history["g_loss"].append(g_epoch_loss)
            self.history["d_loss"].append(d_epoch_loss)
            self.history["epoch_sec"].append(epoch_sec)
            self.history["g_lr"].append(g_lr)
            self.history["d_lr"].append(d_lr)

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar("Loss/G", g_epoch_loss, epoch)
                self.writer.add_scalar("Loss/D", d_epoch_loss, epoch)
                self.writer.add_scalar("LR/G", g_lr, epoch)
                self.writer.add_scalar("LR/D", d_lr, epoch)

            # CSV logging
            self.csv_logger.log_row({
                "epoch": epoch,
                "g_loss": g_epoch_loss,
                "d_loss": d_epoch_loss,
                "epoch_sec": epoch_sec,
                "g_lr": g_lr,
                "d_lr": d_lr,
            })

            # Terminal log for epoch
            if self.cfg.log_every_n_steps:
                print("\r", end="")  # clear step log line
            print(f"Epoch {epoch:03d} | time={epoch_sec:.2f}s | G_loss={g_epoch_loss:.4f} | D_loss={d_epoch_loss:.4f} | G_lr={g_lr:.2e} D_lr={d_lr:.2e}")

            # save generated images
            if self.cfg.plot_every_n_epochs and epoch % self.cfg.plot_every_n_epochs == 0:
                with torch.no_grad():
                    samples = self.G(fixed_z) if fixed_labels is None else self.G(fixed_z, fixed_labels)
                plot_multiple_images(samples, 8, os.path.join(
                    self.cfg.out_dir, "images", f"{self.model_name}_{self.dataset_name}", f"epoch_{epoch}.png"
                ))

            # Early stopping / best checkpoint
            if self.es.monitor == "g_loss":
                monitored_metric = g_epoch_loss
            else:  # "d_loss"
                monitored_metric = d_epoch_loss
            if epoch >= self.es.start_from_epoch:
                if self._is_better(monitored_metric, self.best_metric):
                    self.best_metric = monitored_metric
                    self.best_epoch = epoch
                    self.no_improve_epochs = 0
                    self._save_checkpoint(epoch)
                else:
                    self.no_improve_epochs += 1

                if self.es.patience > 0 and self.no_improve_epochs >= self.es.patience:
                    print(f"Early stopping at epoch {epoch} (best @ {self.best_epoch} = {self.best_metric:.4f}).")
                    break

        # Finalize
        self._plot_curves()
        if self.writer:
            self.writer.close()

        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "checkpoint_path": self.ckpt_path if self.best_epoch > 0 else None
        }
