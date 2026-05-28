# SPDX-License-Identifier: BSD-3-Clause
"""_old3s_classifier.py - OLD³S Classifier for OpenMOA (Optimized Lifelong Logic)"""
from __future__ import annotations
from typing import Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from openmoa.base import Classifier
from openmoa.stream import Schema
from openmoa.instance import Instance

# ======================== Neural Network Components ========================

class VAE_Shallow(nn.Module):
    """Variational Autoencoder for feature extraction."""
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() 
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return z, recon, mu, logvar

class HBPMLP(nn.Module):
    """Multi-exit MLP with Hedge Backpropagation."""
    def __init__(self, input_dim: int, num_classes: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        hidden_dim = 64 
        
        # Layer 1
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        self.classifiers.append(nn.Linear(hidden_dim, num_classes))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
            self.classifiers.append(nn.Linear(hidden_dim, num_classes))
            
    def forward(self, x):
        preds = []
        feat = x
        for i, layer in enumerate(self.layers):
            feat = layer(feat)
            preds.append(self.classifiers[i](feat))
        return preds

# ======================== OLD³S Classifier ========================

class OLD3SClassifier(Classifier):
    """
    OLD³S (Online Learning Deep models from Data of Double Streams).
    
    This implementation supports Lifelong Learning (multiple transitions) by 
    automatically detecting feature space shifts and aligning latent spaces.
    
    Features:
    - Supports infinite sequence of feature spaces (S1 -> S2 -> S3 ...).
    - Reactive Drift Detection: Automatically detects Overlap/Shift based on feature indices.
    - Knowledge Distillation: Aligns new latent space with the previous one.
    
    Reference: Lian, H., et al. (2024). IEEE TKDE.
    """
    
    def __init__(
        self,
        schema: Schema,
        latent_dim: int = 20,
        hidden_dim: int = 128,
        num_hbp_layers: int = 3,
        learning_rate: float = 0.001,
        beta: float = 0.99,      # HBP decay rate
        eta: float = 0.01,       # Ensemble update rate
        random_seed: int = 1,
    ):
        super().__init__(schema=schema, random_seed=random_seed)
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_hbp_layers = num_hbp_layers
        self.lr = learning_rate
        self.beta = beta
        self.eta = eta
        self.num_classes = schema.get_num_classes()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(random_seed)
        self._rng = np.random.RandomState(random_seed)
        
        # === Lifelong State Management ===
        # We maintain only TWO models at any time: 
        # 1. Prev (The teacher/regularizer)
        # 2. Curr (The student/active learner)
        self.model_curr = None
        self.model_prev = None
        
        # Ensemble weights for Curr and Prev
        self.w_curr = 1.0
        self.w_prev = 0.0
        
        # Feature Space Tracking
        self.curr_indices = None    # Indices of current stable space
        self.prev_indices = None    # Indices of previous stable space
        self.is_overlap = False
        
        # Statistics for Normalization
        self.stats_curr = {'min': None, 'max': None}
        self.stats_prev = {'min': None, 'max': None}
        
        # Loss History for Ensemble
        self.loss_hist_curr = deque(maxlen=50)
        self.loss_hist_prev = deque(maxlen=50)
        
        self.t = 0

    def __str__(self):
        return f"OLD3S(latent={self.latent_dim}, lr={self.lr})"

    def _create_model_bundle(self, input_dim):
        """Factory to create VAE + Classifier + Optimizers."""
        vae = VAE_Shallow(input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        clf = HBPMLP(self.latent_dim, self.num_classes, self.num_hbp_layers).to(self.device)
        
        return {
            'vae': vae,
            'clf': clf,
            'opt_vae': torch.optim.Adam(vae.parameters(), lr=self.lr),
            'opt_clf': torch.optim.Adam(clf.parameters(), lr=self.lr),
            'hbp_weights': torch.ones(self.num_hbp_layers, device=self.device) / self.num_hbp_layers,
            'dim': input_dim
        }

    def _normalize(self, x_raw, indices, stats):
        """Online Min-Max normalisation to [0, 1].

        *stats* is a dict with keys ``'min'`` / ``'max'``, each of length
        ``len(indices)`` (the subset dimension), **not** ``len(x_raw)``.
        """
        x_sub = x_raw[indices]

        if stats['min'] is None:
            stats['min'] = x_sub.copy()
            stats['max'] = x_sub.copy()
        else:
            if len(x_sub) == len(stats['min']):   # ← was: len(x_raw), which is wrong
                stats['min'] = np.minimum(stats['min'], x_sub)
                stats['max'] = np.maximum(stats['max'], x_sub)
            else:
                # Dimension mismatch after a feature-space transition;
                # fall back to clipping until stats are re-initialised.
                return np.clip(x_sub, 0.0, 1.0)

        denom = stats['max'] - stats['min']
        denom[denom < 1e-9] = 1.0

        return np.clip((x_sub - stats['min']) / denom, 0.0, 1.0)

    def _detect_stage(self, indices):
        """Reactive State Machine for Feature Evolution."""
        indices_set = set(indices)
        
        if self.model_curr is None:
            self.curr_indices = indices
            self.model_curr = self._create_model_bundle(len(indices))
            self.stats_curr = {'min': None, 'max': None}
            return "STABLE"

        curr_set = set(self.curr_indices)
        
        # Check for Overlap (Superset)
        if indices_set > curr_set:
            if not self.is_overlap:
                # Transition to Overlap
                self.model_prev = self.model_curr 
                self.prev_indices = self.curr_indices
                self.stats_prev = self.stats_curr
                
                # Identify S2 (New features)
                s2_idx_list = sorted(list(indices_set - curr_set))
                if not s2_idx_list: s2_idx_list = indices 
                
                self.curr_indices = np.array(s2_idx_list)
                self.model_curr = self._create_model_bundle(len(self.curr_indices))
                self.stats_curr = {'min': None, 'max': None} 
                
                self.is_overlap = True
                self.w_prev = 0.5
                self.w_curr = 0.5
                
            return "OVERLAP"

        # Check for Shift to New Stable (Subset of Overlap, Disjoint from Old)
        if self.is_overlap and len(indices) < (len(self.prev_indices) + len(self.curr_indices)):
             self.is_overlap = False
             self.model_prev = None 
             return "STABLE_NEW"
             
        return "STABLE"

    def train(self, instance: Instance):
        self.t += 1
        x_full = np.asarray(instance.x, dtype=np.float32)
        x_full = np.nan_to_num(x_full, nan=0.0)
        y = torch.tensor([instance.y_index], dtype=torch.long, device=self.device)
        
        indices = getattr(instance, 'feature_indices', np.arange(len(x_full)))
        
        # 1. Detect Stage
        stage = self._detect_stage(indices)
        
        # 2. Prepare Data Helper
        def get_sub_tensor(model_indices, stats):
            global_to_local = {idx: i for i, idx in enumerate(indices)}
            local_indices = []
            for idx in model_indices:
                if idx in global_to_local:
                    local_indices.append(global_to_local[idx])
            
            if not local_indices: return None
            x_sub = x_full[local_indices]
            x_norm = self._normalize(x_sub, list(range(len(x_sub))), stats)
            return torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 3. Execution
        if stage == "STABLE":
            x_in = get_sub_tensor(self.curr_indices, self.stats_curr)
            if x_in is not None:
                self._train_bundle(self.model_curr, x_in, y)
                
        elif stage == "STABLE_NEW":
            x_in = get_sub_tensor(self.curr_indices, self.stats_curr)
            if x_in is not None:
                self._train_bundle(self.model_curr, x_in, y)
            if self.model_prev is not None and x_in is not None:
                 self._update_ensemble_logic(x_in, y)

        elif stage == "OVERLAP":
            x_prev = get_sub_tensor(self.prev_indices, self.stats_prev)
            if x_prev is not None:
                self._train_bundle(self.model_prev, x_prev, y)
            
            x_curr = get_sub_tensor(self.curr_indices, self.stats_curr)
            if x_curr is not None and x_prev is not None:
                # Alignment
                with torch.no_grad():
                    z_prev, _, _, _ = self.model_prev['vae'](x_prev)
                
                # Custom Train with Alignment
                z_curr, recon, mu, logvar = self.model_curr['vae'](x_curr)
                rec_loss = F.binary_cross_entropy(recon, x_curr, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                align_loss = F.mse_loss(z_curr, z_prev.detach())
                
                total_loss = rec_loss + 0.1 * kld_loss + 10.0 * align_loss
                
                self.model_curr['opt_vae'].zero_grad()
                total_loss.backward()
                self.model_curr['opt_vae'].step()
                
                self._train_clf_only(self.model_curr, z_curr.detach(), y)
                self._update_ensemble_logic(x_curr, y)

    def predict_proba(self, instance: Instance) -> np.ndarray:
        x_full = np.asarray(instance.x, dtype=np.float32)
        indices = getattr(instance, 'feature_indices', np.arange(len(x_full)))
        
        if self.model_curr is None:
            return np.ones(self.num_classes) / self.num_classes

        global_to_local = {idx: i for i, idx in enumerate(indices)}
        local_indices = [global_to_local[idx] for idx in self.curr_indices if idx in global_to_local]
        
        if not local_indices: return np.ones(self.num_classes) / self.num_classes
        
        x_sub = x_full[local_indices]
        x_tensor = torch.tensor(x_sub, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            z_curr, _, _, _ = self.model_curr['vae'](x_tensor)
            logits_curr = self._predict_hbp(self.model_curr, z_curr)
            
            logits_final = logits_curr
            
            if self.model_prev is not None:
                logits_prev = self._predict_hbp(self.model_prev, z_curr)
                logits_final = self.w_curr * logits_curr + self.w_prev * logits_prev
                
        return F.softmax(logits_final, dim=1).cpu().numpy()[0]
        
    def predict(self, instance: Instance) -> int:
        return np.argmax(self.predict_proba(instance))

    def _train_bundle(self, bundle, x, y):
        z, recon, mu, logvar = bundle['vae'](x)
        loss = F.binary_cross_entropy(recon, x, reduction='sum') - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        bundle['opt_vae'].zero_grad()
        loss.backward()
        bundle['opt_vae'].step()
        self._train_clf_only(bundle, z.detach(), y)

    def _train_clf_only(self, bundle, z, y):
        preds = bundle['clf'](z)
        losses = [F.cross_entropy(p, y) for p in preds]
        w_loss = sum(w * l for w, l in zip(bundle['hbp_weights'], losses))
        bundle['opt_clf'].zero_grad()
        w_loss.backward()
        bundle['opt_clf'].step()
        with torch.no_grad():
            # P4: vectorised in-place update — no per-layer Python loop or temp tensors
            decay = torch.tensor(self.beta, device=self.device)
            losses_t = torch.stack([l.detach() for l in losses])  # (num_layers,)
            bundle['hbp_weights'].mul_(decay.pow(losses_t))
            bundle['hbp_weights'].div_(bundle['hbp_weights'].sum())

    def _predict_hbp(self, bundle, z):
        preds = bundle['clf'](z)
        return sum(w * p for w, p in zip(bundle['hbp_weights'], preds))

    def _update_ensemble_logic(self, x_curr, y):
        with torch.no_grad():
            z, _, _, _ = self.model_curr['vae'](x_curr)
            l_c = F.cross_entropy(self._predict_hbp(self.model_curr, z), y).item()
            l_p = F.cross_entropy(self._predict_hbp(self.model_prev, z), y).item()
        self.loss_hist_curr.append(l_c)
        self.loss_hist_prev.append(l_p)
        
        avg_c = np.mean(self.loss_hist_curr)
        avg_p = np.mean(self.loss_hist_prev)
        
        w_c = np.exp(-self.eta * avg_c)
        w_p = np.exp(-self.eta * avg_p)
        total = w_c + w_p + 1e-9
        self.w_curr = w_c / total
        self.w_prev = w_p / total