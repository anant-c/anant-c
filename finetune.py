import gc
import os
import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

# Clear memory
gc.collect()
torch.cuda.empty_cache()

# Load model
model = EncDecSpeakerLabelModel.from_pretrained("ecapa_tdnn")
cfg = model.cfg

# Configure dataset with longer segments
OmegaConf.set_struct(cfg, False)
cfg.train_ds.manifest_filepath = "train_manifest.json"
cfg.train_ds.batch_size = 2  # Try batch size of 2
cfg.train_ds.num_workers = 2
cfg.train_ds.shuffle = True
if 'augmentor' in cfg.train_ds:
    cfg.train_ds.augmentor = None
if not hasattr(cfg.train_ds, 'min_duration'):
    cfg.train_ds.min_duration = 3.0  # Ensure longer segments
OmegaConf.set_struct(cfg, True)

# Modify batch norm layers to work with small batches
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm1d):
        module.momentum = 0.1
        module.track_running_stats = False

# Set up training data
model = model.to("cpu")
model.setup_training_data(cfg.train_ds)

# Configure trainer
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    precision=16,
    max_epochs=15,
    accumulate_grad_batches=2,
    log_every_n_steps=1
)

# Train model
model = model.to("cuda")
trainer.fit(model)
