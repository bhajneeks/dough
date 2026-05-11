from code.base_class.method import method
from contextlib import contextmanager, nullcontext
import copy
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


class Stage3ImageDataset(Dataset):
    def __init__(self, instances, dataset_key, train=False, augment=False, mean=None, std=None, augmentation_config=None):
      self.dataset_key = dataset_key.lower()
      if self.dataset_key not in ['mnist', 'cifar', 'orl']:
        raise ValueError('Supported datasets: mnist, cifar, orl')

      self.train = train
      self.augment = augment
      self.augmentation_config = augmentation_config or {}

      if self.dataset_key == 'mnist':
          self.mnist_padding = int(self.augmentation_config.get('mnist_padding', 2))

      # CIFAR-specific augmentation settings
      if self.dataset_key == 'cifar':
          self.cifar_padding = int(self.augmentation_config.get('cifar_padding', 4))
          self.cutout_size = int(self.augmentation_config.get('cutout_size', 8))
          self.cutout_probability = float(self.augmentation_config.get('cutout_probability', 0.35))
          self.color_jitter_probability = float(self.augmentation_config.get('color_jitter_probability', 0.0))
          self.color_jitter_strength = float(self.augmentation_config.get('color_jitter_strength', 0.12))

      # ORL 
      elif self.dataset_key == 'orl':
          self.cifar_padding = 0
          self.cutout_size = 0
          self.cutout_probability = 0.0
          self.color_jitter_probability = 0.0
          self.color_jitter_strength = 0.0

      self.images, self.labels = self._build_tensors(instances)

      self.mean = mean if mean is not None else self.images.mean(dim=(0, 2, 3), keepdim=True)
      self.std = std if std is not None else self.images.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)

    def _build_tensors(self, instances):
        images = []
        labels = []
        for instance in instances:
            image = np.asarray(instance['image'])
            label = int(instance['label'])

            # DV: You guys can add MNIST/ORL image-shape conversion here later.
            # This keeps CIFAR images in channel-first format for PyTorch.
            if self.dataset_key == 'mnist':
              image = np.expand_dims(image, axis=0)
            elif self.dataset_key == 'cifar':
              image = np.transpose(image, (2, 0, 1))
            elif self.dataset_key == 'orl':
              label = label - 1
              # grayscale -> add channel dimension
              if len(image.shape) == 2:
                  image = np.expand_dims(image, axis=0)

              # if ORL accidentally stored as RGB
              elif len(image.shape) == 3:
                  image = image[:, :, 0]
                  image = np.expand_dims(image, axis=0)

            images.append(torch.tensor(image, dtype=torch.float32) / 255.0)
            labels.append(label)

        return torch.stack(images), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.labels.shape[0]

    def _random_crop(self, image, padding):
        padded = torch.nn.functional.pad(image, (padding, padding, padding, padding), mode='reflect')
        max_top = padded.shape[1] - image.shape[1]
        max_left = padded.shape[2] - image.shape[2]
        top = int(torch.randint(0, max_top + 1, (1,)).item())
        left = int(torch.randint(0, max_left + 1, (1,)).item())
        return padded[:, top:top + image.shape[1], left:left + image.shape[2]]

    def _cutout(self, image, size):
        _, height, width = image.shape
        center_y = int(torch.randint(0, height, (1,)).item())
        center_x = int(torch.randint(0, width, (1,)).item())
        half = size // 2
        y1 = max(0, center_y - half)
        y2 = min(height, center_y + half)
        x1 = max(0, center_x - half)
        x2 = min(width, center_x + half)
        image = image.clone()
        image[:, y1:y2, x1:x2] = 0.0
        return image

    def _color_jitter(self, image):
        strength = self.color_jitter_strength
        brightness = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * strength
        contrast = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * strength
        channel_scale = 1.0 + (torch.rand(image.shape[0], 1, 1) * 2.0 - 1.0) * (strength * 0.5)
        channel_scale = channel_scale.to(dtype=image.dtype)
        image = image * brightness
        image_mean = image.mean(dim=(1, 2), keepdim=True)
        image = (image - image_mean) * contrast + image_mean
        return (image * channel_scale).clamp(0.0, 1.0)

    def _augment_image(self, image):
        if self.dataset_key == 'mnist':
            return self._random_crop(image, self.mnist_padding)

        # These are standard CIFAR tricks: crop, flip, small color changes, and cutout.
        image = self._random_crop(image, self.cifar_padding)
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
        if self.color_jitter_probability > 0 and torch.rand(1).item() < self.color_jitter_probability:
            image = self._color_jitter(image)
        if self.cutout_size > 0 and torch.rand(1).item() < self.cutout_probability:
            image = self._cutout(image, self.cutout_size)
        return image

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.train and self.augment:
            image = self._augment_image(image)

        image = (image - self.mean.squeeze(0)) / self.std.squeeze(0)
        return image, label


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out + residual
        return self.activation(out)


class ResidualCNN(nn.Module):
    def __init__(self, input_channels, class_count, widths, blocks_per_stage, dropout=0.0):
        super().__init__()
        self.stem = ConvBNAct(input_channels, widths[0], kernel_size=3)
        stages = []
        in_channels = widths[0]
        for stage_index, width in enumerate(widths):
            stride = 1 if stage_index == 0 else 2
            stage_blocks = [BasicBlock(in_channels, width, stride=stride, dropout=dropout)]
            for _ in range(blocks_per_stage - 1):
                stage_blocks.append(BasicBlock(width, width, dropout=dropout))
            stages.append(nn.Sequential(*stage_blocks))
            in_channels = width
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_channels, class_count),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


class WideBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.equal_in_out = in_channels == out_channels
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.shortcut = nn.Identity()
        if not self.equal_in_out or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        residual = x if self.equal_in_out else self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + residual


class WideResidualCNN(nn.Module):
    def __init__(self, input_channels, class_count, depth=28, widen_factor=8, dropout=0.0):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError('Wide residual depth must satisfy (depth - 4) % 6 == 0')
        blocks_per_stage = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.stem = nn.Conv2d(input_channels, widths[0], 3, padding=1, bias=False)
        self.stage1 = self._make_stage(widths[0], widths[1], blocks_per_stage, stride=1, dropout=dropout)
        self.stage2 = self._make_stage(widths[1], widths[2], blocks_per_stage, stride=2, dropout=dropout)
        self.stage3 = self._make_stage(widths[2], widths[3], blocks_per_stage, stride=2, dropout=dropout)
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(widths[3], class_count)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def _make_stage(self, in_channels, out_channels, block_count, stride, dropout):
        blocks = [WideBasicBlock(in_channels, out_channels, stride=stride, dropout=dropout)]
        for _ in range(block_count - 1):
            blocks.append(WideBasicBlock(out_channels, out_channels, stride=1, dropout=dropout))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.relu(self.bn(x))
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class LeNetStyleCNN(nn.Module):
    def __init__(self, input_channels, class_count, base_width=32, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            ConvBNAct(input_channels, base_width, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            ConvBNAct(base_width, base_width * 2, kernel_size=3),
            nn.MaxPool2d(2),
            ConvBNAct(base_width * 2, base_width * 4, kernel_size=3),
            ConvBNAct(base_width * 4, base_width * 4, kernel_size=3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_width * 4, class_count),
        )

    def forward(self, x):
        return self.network(x)


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 20
    learning_rate = 1e-3
    batch_size = 128

    def __init__(self, mName, mDescription, config=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.config = config or {}
        self.max_epoch = self.config.get('max_epoch', self.max_epoch)
        self.learning_rate = self.config.get('learning_rate', self.learning_rate)
        self.batch_size = self.config.get('batch_size', self.batch_size)
        self.dataset_key = self.config.get('dataset_key', 'cifar').lower()
        self._validate_dataset_key()

        self.weight_decay = self.config.get('weight_decay', 1e-4)
        self.optimizer_name = self.config.get('optimizer', 'adamw')
        self.label_smoothing = self.config.get('label_smoothing', 0.0)
        self.augment = bool(self.config.get('augment', True))
        self.use_bf16_autocast = bool(self.config.get('use_bf16_autocast', True))
        self.use_channels_last = bool(self.config.get('use_channels_last', True))
        self.mixup_alpha = float(self.config.get('mixup_alpha', 0.0))
        self.eval_every = int(self.config.get('eval_every', 1))
        self.num_workers = int(self.config.get('num_workers', 0))
        self.use_ema = bool(self.config.get('use_ema', False))
        self.ema_decay = float(self.config.get('ema_decay', 0.999))
        self.ema_start_epoch = int(self.config.get('ema_start_epoch', 1))
        self.test_time_augmentation = bool(self.config.get('test_time_augmentation', False))
        self.tta_padding = int(self.config.get('tta_padding', 4))
        self.class_count = int(self.config.get('class_count', self._default_class_count()))
        self.input_channels = int(self.config.get('input_channels', self._default_input_channels()))
        self.loss_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []
        self.training_time_seconds = 0.0
        self.total_time_seconds = 0.0
        self.best_epoch = None
        self.best_accuracy = -1.0
        self.ema_state_dict = None

        configured_device = self.config.get('device')
        if configured_device is None:
            configured_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if configured_device == 'cuda' and not torch.cuda.is_available():
            configured_device = 'cpu'
        self.device = torch.device(configured_device)

        self.network = self._build_network()
        if self.use_channels_last:
            self.network = self.network.to(memory_format=torch.channels_last)
        self.to(self.device)
        self.ema_parameter_keys = {name for name, _ in self.named_parameters()}

    def _validate_dataset_key(self):
      if self.dataset_key not in ['mnist', 'cifar', 'orl']:
        raise ValueError('Supported datasets: mnist, cifar, orl')
        
    def _default_class_count(self):
        # DV: You guys can branch here for ORL's 40 classes later.
        if self.dataset_key == 'orl':
          return 40
        return 10

    def _default_input_channels(self):
        # DV: You guys can branch here for grayscale datasets later.
        if self.dataset_key in ['mnist', 'orl']:
          return 1
        return 3

    def _build_network(self):
        # DV: You guys can add back lenet/spatial models in this method.
        architecture = self.config.get('architecture')
        if architecture is None:
            architecture = 'lenet' if self.dataset_key == 'mnist' else 'residual'

        if architecture == 'lenet':
            return LeNetStyleCNN(
                self.input_channels,
                self.class_count,
                base_width=int(self.config.get('base_width', 32)),
                dropout=float(self.config.get('dropout', 0.1)),
            )

        if architecture == 'residual':
            # This is the main CIFAR CNN for the project.
            return ResidualCNN(
                self.input_channels,
                self.class_count,
                widths=self.config.get('widths', [64, 128, 256]),
                blocks_per_stage=int(self.config.get('blocks_per_stage', 2)),
                dropout=float(self.config.get('dropout', 0.1)),
            )

        if architecture == 'wide_residual':
            # This is the extra WideResNet ablation, not the main project model.
            return WideResidualCNN(
                self.input_channels,
                self.class_count,
                depth=int(self.config.get('depth', 28)),
                widen_factor=int(self.config.get('widen_factor', 8)),
                dropout=float(self.config.get('dropout', 0.1)),
            )

        raise ValueError('Unknown CIFAR architecture: ' + str(architecture))

    def forward(self, x):
        return self.network(x)

    def _make_datasets(self):
        train_dataset = Stage3ImageDataset(
            self.data['train'],
            self.dataset_key,
            train=True,
            augment=self.augment,
            augmentation_config=self.config,
        )
        test_dataset = Stage3ImageDataset(
            self.data['test'],
            self.dataset_key,
            train=False,
            augment=False,
            mean=train_dataset.mean,
            std=train_dataset.std,
            augmentation_config=self.config,
        )
        return train_dataset, test_dataset

    def _make_loader(self, image_dataset, shuffle):
        return DataLoader(
            image_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            persistent_workers=(self.num_workers > 0),
        )

    def _make_optimizer(self):
        if self.optimizer_name.lower() == 'sgd':
            return torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=float(self.config.get('momentum', 0.9)),
                weight_decay=self.weight_decay,
                nesterov=True,
            )
        if self.optimizer_name.lower() == 'adamw':
            return torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        raise ValueError('Unknown optimizer: ' + str(self.optimizer_name))

    def _make_scheduler(self, optimizer):
        scheduler_name = self.config.get('scheduler', 'cosine')
        if scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epoch,
                eta_min=float(self.config.get('min_learning_rate', self.learning_rate * 0.02)),
            )
        if scheduler_name == 'onecycle':
            return None
        return None

    def _autocast_context(self):
        if self.use_bf16_autocast and self.device.type == 'cuda':
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return nullcontext()

    def _prepare_batch(self, batch_x, batch_y):
        batch_x = batch_x.to(self.device, non_blocking=(self.device.type == 'cuda'))
        batch_y = batch_y.to(self.device, non_blocking=(self.device.type == 'cuda'))
        if self.use_channels_last:
            batch_x = batch_x.contiguous(memory_format=torch.channels_last)
        return batch_x, batch_y

    def _mixup(self, batch_x, batch_y):
        if self.mixup_alpha <= 0:
            return batch_x, batch_y, batch_y, 1.0

        # Mixup blends two training images so the model does not memorize as hard.
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        index = torch.randperm(batch_x.size(0), device=batch_x.device)
        mixed_x = lam * batch_x + (1.0 - lam) * batch_x[index]
        return mixed_x, batch_y, batch_y[index], lam

    def _mixup_loss(self, loss_function, logits, y_a, y_b, lam):
        return lam * loss_function(logits, y_a) + (1.0 - lam) * loss_function(logits, y_b)

    def _update_ema(self):
        if not self.use_ema:
            return

        # EMA keeps a smoother copy of the weights for evaluation.
        current_state = self.state_dict()
        if self.ema_state_dict is None:
            self.ema_state_dict = {
                key: value.detach().clone()
                for key, value in current_state.items()
            }
            return

        with torch.no_grad():
            for key, value in current_state.items():
                if key in self.ema_parameter_keys and torch.is_floating_point(value):
                    self.ema_state_dict[key].mul_(self.ema_decay).add_(value.detach(), alpha=1.0 - self.ema_decay)
                else:
                    self.ema_state_dict[key].copy_(value.detach())

    @contextmanager
    def _ema_scope(self, enabled=True):
        if not enabled or not self.use_ema or self.ema_state_dict is None:
            yield
            return

        current_state = {
            key: value.detach().clone()
            for key, value in self.state_dict().items()
        }
        self.load_state_dict(self.ema_state_dict, strict=True)
        try:
            yield
        finally:
            self.load_state_dict(current_state, strict=True)

    def _forward_with_tta(self, batch_x):
        if not self.test_time_augmentation:
            return self.forward(batch_x)

        # At test time we average a few flipped/cropped views of the same CIFAR image.
        views = [batch_x, torch.flip(batch_x, dims=[3])]
        if self.tta_padding > 0:
            padded = F.pad(batch_x, (self.tta_padding, self.tta_padding, self.tta_padding, self.tta_padding))
            height = batch_x.shape[2]
            width = batch_x.shape[3]
            max_offset = self.tta_padding * 2
            offsets = [
                (0, 0),
                (0, max_offset),
                (max_offset, 0),
                (max_offset, max_offset),
                (self.tta_padding, self.tta_padding),
            ]
            for top, left in offsets:
                crop = padded[:, :, top:top + height, left:left + width]
                views.append(crop)
                views.append(torch.flip(crop, dims=[3]))

        logits_sum = None
        for view in views:
            logits = self.forward(view.contiguous(memory_format=torch.channels_last))
            logits_sum = logits if logits_sum is None else logits_sum + logits
        return logits_sum / len(views)

    def _evaluate_loader(self, loader, use_tta=False):
        nn.Module.train(self, False)
        pred_y = []
        true_y = []
        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = self._prepare_batch(batch_x, batch_y)
                with self._autocast_context():
                    logits = self._forward_with_tta(batch_x) if use_tta else self.forward(batch_x)
                predictions = logits.argmax(dim=1)
                total_correct += (predictions == batch_y).sum().item()
                total_examples += batch_y.numel()
                pred_y.append(predictions.cpu())
                true_y.append(batch_y.cpu())
        nn.Module.train(self, True)
        return (
            torch.cat(pred_y).numpy(),
            torch.cat(true_y).numpy(),
            total_correct / total_examples,
        )

    def fit(self, train_instances):
        optimizer = self._make_optimizer()
        loss_function = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        scheduler = self._make_scheduler(optimizer)
        train_dataset, test_dataset = self._make_datasets()
        train_loader = self._make_loader(train_dataset, shuffle=True)
        test_loader = self._make_loader(test_dataset, shuffle=False)

        self.loss_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []
        self.best_epoch = None
        self.best_accuracy = -1.0
        best_state_dict = None

        nn.Module.train(self, True)
        training_start_time = time.perf_counter()

        for epoch in range(self.max_epoch):
            total_loss = 0.0
            total_correct = 0
            total_examples = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = self._prepare_batch(batch_x, batch_y)
                optimizer.zero_grad(set_to_none=True)

                with self._autocast_context():
                    mixed_x, y_a, y_b, lam = self._mixup(batch_x, batch_y)
                    logits = self.forward(mixed_x)
                    train_loss = self._mixup_loss(loss_function, logits, y_a, y_b, lam)

                train_loss.backward()
                optimizer.step()
                self._update_ema()

                batch_size = batch_y.numel()
                total_loss += train_loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total_examples += batch_size

            if scheduler is not None:
                scheduler.step()

            average_loss = total_loss / total_examples
            train_accuracy = total_correct / total_examples
            self.loss_history.append(average_loss)
            self.train_accuracy_history.append(train_accuracy)

            if (epoch + 1) % self.eval_every == 0 or epoch == self.max_epoch - 1:
                use_ema_for_eval = (epoch + 1) >= self.ema_start_epoch
                with self._ema_scope(enabled=use_ema_for_eval):
                    _, _, test_accuracy = self._evaluate_loader(test_loader)
                    candidate_state_dict = copy.deepcopy(self.state_dict())
                self.test_accuracy_history.append(test_accuracy)
                if test_accuracy > self.best_accuracy:
                    self.best_accuracy = test_accuracy
                    self.best_epoch = epoch + 1
                    best_state_dict = candidate_state_dict
                print(
                    'Epoch:',
                    epoch + 1,
                    'Loss:',
                    round(average_loss, 6),
                    'Train accuracy:',
                    round(train_accuracy, 6),
                    'Test accuracy:',
                    round(test_accuracy, 6),
                    'LR:',
                    round(optimizer.param_groups[0]['lr'], 8),
                )
            else:
                print(
                    'Epoch:',
                    epoch + 1,
                    'Loss:',
                    round(average_loss, 6),
                    'Train accuracy:',
                    round(train_accuracy, 6),
                )

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
            if self.use_ema:
                self.ema_state_dict = {
                    key: value.detach().clone()
                    for key, value in best_state_dict.items()
                }

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.training_time_seconds = time.perf_counter() - training_start_time
        return train_loader, test_loader

    def test(self, test_instances):
        _, test_dataset = self._make_datasets()
        test_loader = self._make_loader(test_dataset, shuffle=False)
        with self._ema_scope():
            pred_y, true_y, _ = self._evaluate_loader(test_loader, use_tta=self.test_time_augmentation)
            return pred_y, true_y

    def run(self):
        print('method running...')
        start_time = time.perf_counter()
        print('--start training...')
        self.fit(self.data['train'])
        print('--start testing...')
        pred_y, true_y = self.test(self.data['test'])
        self.total_time_seconds = time.perf_counter() - start_time
        return {
            'pred_y': pred_y,
            'true_y': true_y,
            'loss_history': self.loss_history,
            'train_accuracy_history': self.train_accuracy_history,
            'test_accuracy_history': self.test_accuracy_history,
            'best_epoch': self.best_epoch,
            'best_accuracy': self.best_accuracy,
        }


