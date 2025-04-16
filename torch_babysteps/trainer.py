import torch
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import time
import pandas as pd
from torch.optim.lr_scheduler import _LRScheduler

def get_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available . Using CPU.")

    return device

def train_classifier(model, 
                train_ds,
                loss_function,
                optimizer, 
                device,
                num_epochs = 5,
                **kwargs):
    
    start_time = time.time()
    model.to(device)
    model.train()
    epoch_loss = []
    epoch_time = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(train_ds)):

            inputs = inputs.to(device)
            labels = labels.to(device)

            ##forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            ##backward pass
            optimizer.zero_grad() # clear previous gradients
            loss.backward()  # compute gradients
            optimizer.step() # Update weights

            running_loss += loss.item()
            # if (i+1)% 100 == 0:
                # print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        avg_loss = running_loss/len(train_ds)
        epoch_loss += [avg_loss]
        epoch_end_time = time.time()
        epoch_time += [epoch_end_time - epoch_start_time]
        print(f'---------- Epoch {epoch+1} Finished ---------- Average Loss: {avg_loss:.4f}')

    end_time = time.time()
    print(f"----Training Finished-----")
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")
    train_history = pd.DataFrame(
        {'epoch':range(num_epochs),
        'loss': epoch_loss,
        'training_time': epoch_time}
    )
    return train_history

class FashionMNISTClassifier(nn.Module):
    def __init__(self):
        super(FashionMNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Learning rate scheduler that combines linear warmup and cosine annealing.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps for linear warmup.
        total_steps (int): Total number of training steps.
        min_lr (float): Minimum learning rate after cosine annealing. Default: 0.0.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Behavior:
    - The learning rate increases linearly from a small value (effectively 0) to the base LR
      set in the optimizer over `warmup_steps`.
    - After `warmup_steps`, the learning rate decreases following a cosine curve
      from the base LR down to `min_lr` over the remaining steps (`total_steps` - `warmup_steps`).
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)

        # Input validation
        if not warmup_steps >= 0:
             raise ValueError("Warmup steps must be non-negative.")
        if not total_steps >= warmup_steps:
            raise ValueError("Total steps must be greater than or equal to warmup steps.")
        if not min_lr >= 0.0:
             raise ValueError("Minimum learning rate must be non-negative.")

        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate using chainable form of the scheduler."""
        # self.last_epoch is the current step count (starts at 0)
        # self.base_lrs is the initial learning rate list set in the optimizer

        # if not self._get_lr_called_within_step:
        #     warnings.warn("To get the learning rate computed by the scheduler, "
        #                   "please use `get_last_lr()`.", UserWarning)

        current_step = self.last_epoch + 1 # Use 1-based indexing for calculations

        # --- Warmup Phase ---
        if current_step <= self.warmup_steps:
            if self.warmup_steps == 0: # Handle edge case
                 warmup_factor = 1.0
            else:
                 # Linear increase from 0 to 1
                 warmup_factor = float(current_step) / float(self.warmup_steps)
            # Scale base_lr by the warmup factor
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # --- Cosine Annealing Phase ---
        else:
            # Calculate progress in the cosine phase (from 0 to 1)
            steps_after_warmup = current_step - self.warmup_steps
            total_cosine_steps = self.total_steps - self.warmup_steps

            if total_cosine_steps <= 0: # Handle edge case if total_steps == warmup_steps
                 # Stay at base_lr if no cosine steps are planned
                 return list(self.base_lrs)

            cosine_progress = float(steps_after_warmup) / float(total_cosine_steps)
            # Ensure progress doesn't exceed 1.0 (can happen if total_steps is underestimated)
            cosine_progress = min(cosine_progress, 1.0)

            # Calculate cosine decay factor (from 1 to 0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))

            # Calculate the final learning rate: min_lr + (base_lr - min_lr) * cosine_decay
            # We apply this to each base_lr in the optimizer's param groups
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]
