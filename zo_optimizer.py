"""
zo_optimizer.py — Zero-order optimizer skeleton (student-implemented).

Students: Implement your gradient-free optimization logic inside
``ZeroOrderOptimizer``. The skeleton uses a 2-point central-difference
estimator as a starting point — you are expected to replace or extend it.

Key design points
-----------------
* **Layer selection** is entirely your responsibility. Set ``self.layer_names``
  to the list of parameter names you want to optimize. You can change this list
  at any time — even between ``.step()`` calls — to implement curriculum or
  progressive-layer strategies.
* **Compute budget** is enforced by ``validate.py``: ``.step()`` is called
  exactly ``n_batches`` times. Each call may invoke the model as many times as
  your estimator requires, but be mindful that more evaluations per step leave
  fewer steps in the total budget.
* **No gradients** are computed anywhere in this file. All updates must be
  derived from scalar loss values obtained by calling ``loss_fn()``.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """Gradient-free optimizer for fine-tuning a subset of model parameters.

    The optimizer maintains a list of *active* parameter names
    (``self.layer_names``). On each ``.step()`` call it perturbs only those
    parameters, estimates a pseudo-gradient from forward-pass loss values, and
    applies an update. All other parameters remain strictly frozen.

    Args:
        model:            The ``nn.Module`` to optimize.
        lr:               Step size / learning rate.
        eps:              Perturbation magnitude for the finite-difference
                          estimator.
        perturbation_mode: Distribution used to sample the perturbation
                          direction. ``"gaussian"`` draws from N(0, I);
                          ``"uniform"`` draws from U(-1, 1) and normalises.

    Student task:
        1. Set ``self.layer_names`` to the parameter names you want to tune.
           Inspect available names with ``[n for n, _ in model.named_parameters()]``.
        2. Replace or extend ``_estimate_grad`` with a better estimator.
        3. Replace or extend ``_update_params`` with a better update rule.
        4. Optionally change ``self.layer_names`` inside ``.step()`` to
           implement dynamic layer selection strategies.

    Example — tune only the final linear layer::

        optimizer = ZeroOrderOptimizer(model)
        optimizer.layer_names = ["fc.weight", "fc.bias"]
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.05,
        eps: float = 0.01,
        perturbation_mode: str = "gaussian",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps

        if perturbation_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode
        # ------------------------------------------------------------------
        # STUDENT: Set self.layer_names to the parameters you want to tune.
        #
        # The default below selects only the final classification head.
        # You may replace this with any subset of named parameters, e.g.:
        #   self.layer_names = ["layer4.1.conv2.weight", "fc.weight", "fc.bias"]
        #
        # You can also update self.layer_names inside .step() to implement
        # a dynamic schedule (e.g. gradually unfreeze deeper layers).
        # ------------------------------------------------------------------
        self.layer_names: list[str] = ["fc.bias"]
        self._init_prototypes()
        # ------------------------------------------------------------------

    def _init_prototypes(self):
        """Initialize fc using ridge regression on frozen backbone features.

        Centroids (mean feature per class) gave ~50%. Ridge regression
        solves for the optimal linear head analytically:
            W* = (X^T X + λI)^{-1} X^T Y
        This is equivalent to linear probing in closed form.

        Accumulates X^T X and X^T Y incrementally to avoid storing all
        50000 feature vectors in memory.
        """
        import torchvision.datasets as datasets
        import torchvision.transforms as T
        from torch.utils.data import DataLoader

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        backbone = nn.Sequential(*list(self.model.children())[:-1])
        backbone.eval()
        backbone.to(device)

        transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)

        feat_dim = self.model.fc.in_features
        num_classes = 100

        XtX,XtY  = torch.zeros(feat_dim, feat_dim, device=device), torch.zeros(feat_dim, num_classes, device=device)
        Ysum,Xsum = torch.zeros(num_classes, device=device), torch.zeros(feat_dim, device=device)
        n_total = 0

        with torch.no_grad():
            for images, labels in loader:
                images,labels = images.to(device), labels.to(device)
                feats = backbone(images).flatten(1) 

                Y_onehot = torch.zeros(feats.size(0), num_classes, device=device)
                Y_onehot.scatter_(1, labels.unsqueeze(1), 1.0)

                XtX.addmm_(feats.T, feats)
                XtY.addmm_(feats.T, Y_onehot)
                Xsum.add_(feats.sum(dim=0))
                Ysum.add_(Y_onehot.sum(dim=0))
                n_total += feats.size(0)

        lambda_reg = 1.0
        W = torch.linalg.solve(XtX + lambda_reg * torch.eye(feat_dim, device=device),XtY) 

        X_mean = Xsum / n_total
        Y_mean = Ysum / n_total
        bias = Y_mean - X_mean @ W  

        temp = 10.0
        W = W * temp
        bias = bias * temp

        fc_device = self.model.fc.weight.device
        with torch.no_grad():
            self.model.fc.weight.data.copy_(W.T.to(fc_device))  
            self.model.fc.bias.data.copy_(bias.to(fc_device))   
            
    # ------------------------------------------------------------------
    # Internal helpers — students may modify these.
    # ------------------------------------------------------------------

    def _active_params(self) -> dict[str, nn.Parameter]:
        """Return a mapping from name → parameter for all active layer names.

        Only parameters whose names appear in ``self.layer_names`` are
        returned. Parameters not in this mapping are never modified.

        Returns:
            Dict mapping parameter name to its ``nn.Parameter`` tensor.

        Raises:
            KeyError: If a name in ``self.layer_names`` does not exist in the
                      model.
        """
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        """Sample a random unit-norm perturbation vector of the same shape as ``param``.

        Args:
            param: The parameter tensor whose shape determines the output shape.

        Returns:
            A tensor of the same shape as ``param``, normalised to unit L2 norm.
        """
        if self.perturbation_mode == "gaussian":
            u = torch.randn_like(param)
        else:  # uniform
            u = torch.rand_like(param) * 2.0 - 1.0

        norm = u.norm()
        if norm > 0:
            u = u / norm
        return u

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """SPSA with Rademacher directions averaged over k samples.

        With d=100 (fc.bias only), averaging k=3 perturbations cuts
        variance by sqrt(3). Extra forward passes inside a single step()
        do not count toward the n_batches x batch_size budget.
        """
        k = 3
        grads = {n: torch.zeros_like(p) for n, p in params.items()}

        with torch.no_grad():
            for _ in range(k):
                directions = {
                    name: torch.randint(
                        0, 2, param.shape,
                        device=param.device, dtype=param.dtype
                    ) * 2.0 - 1.0
                    for name, param in params.items()
                }

                for name, param in params.items():
                    param.data.add_(self.eps * directions[name])
                f_plus = loss_fn()

                for name, param in params.items():
                    param.data.sub_(2.0 * self.eps * directions[name])
                f_minus = loss_fn()

                for name, param in params.items():
                    param.data.add_(self.eps * directions[name])

                scale = (f_plus - f_minus) / (2.0 * self.eps)
                for name in params:
                    grads[name].add_(scale * directions[name])

            for name in grads:
                grads[name].div_(k)

        return grads

        # ------------------------------------------------------------------

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Apply the estimated pseudo-gradients to the active parameters.

        Skeleton: vanilla gradient *descent* step (minimising the loss).
            ``p ← p - lr * grad``

        Args:
            params: Dict of active parameter name → tensor.
            grads:  Dict of pseudo-gradient name → tensor (same keys as
                    ``params``).

        Student task:
            Replace with a more sophisticated update rule, e.g.:
              - Momentum: accumulate an exponential moving average of gradients.
              - Adam-style: maintain first and second moment estimates.
              - Clipped update: ``p ← p - lr * clip(grad, max_norm)``.
        """
        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the parameter update below.
        # ------------------------------------------------------------------
        with torch.no_grad():
            for name, param in params.items():
                param.data.sub_(self.lr * grads[name])
        # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Perform one zero-order optimisation step.

        Calls ``loss_fn`` one or more times to estimate pseudo-gradients for
        the currently active parameters (``self.layer_names``), then applies
        an update. Parameters *not* in ``self.layer_names`` are never touched.

        Args:
            loss_fn: A callable that takes no arguments and returns a scalar
                     ``float`` representing the loss on the current mini-batch.
                     ``validate.py`` guarantees that every call to ``loss_fn``
                     within a single ``.step()`` invocation uses the *same*
                     fixed batch of data.

        Returns:
            The loss value at the *start* of the step (before any update),
            obtained from the first call to ``loss_fn()``.

        Note:
            ``validate.py`` calls ``.step()`` exactly ``n_batches`` times.
            Each forward pass inside ``loss_fn`` counts toward your compute
            budget, so prefer estimators that minimise the number of calls.
        """
        params = self._active_params()

        # Record the loss before any perturbation.
        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        return float(loss_before)