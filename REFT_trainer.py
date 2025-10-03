import torch
import torch.nn.functional as F
from pyreft import ReftTrainerForCausalLM
import wandb
import torch.nn as nn
from typing import Optional, Dict, Any



class ReftTrainerImplicit(ReftTrainerForCausalLM):
    def __init__(self, lambda_consistency=0.1, config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_consistency = lambda_consistency
        self.config = config
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Compute the original loss
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False)
        
        total_consistency_loss = 0.0
        num_consistency_interventions = 0
       
        
        
        # Collect consistency losses from LoreftIntervention_Consistency interventions
        if self.lambda_consistency > 0 and hasattr(model, 'interventions'):
            for module_key, intervention in model.interventions.items():
                # Check if this intervention is of type LoreftIntervention_Consistency
                if hasattr(intervention, 'lambda_consistency') and intervention.lambda_consistency > 0:
                    # During forward pass, consistency-enabled interventions store their losses
                    if hasattr(intervention, '_last_consistency_loss') and intervention._last_consistency_loss is not None:
                        total_consistency_loss += intervention._last_consistency_loss
                        num_consistency_interventions += 1
                        # Clear the stored loss to avoid double-counting
                        intervention._last_consistency_loss = None
        
    
        if num_consistency_interventions > 0:
            # Average consistency loss across interventions
            avg_consistency_loss = total_consistency_loss / num_consistency_interventions
            loss = loss + self.lambda_consistency * avg_consistency_loss
            
     
        
        return (loss, outputs) if return_outputs else loss



class ReftTrainerAdv(ReftTrainerForCausalLM):
    def __init__(self, lambda_consistency=0.1, config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_consistency = lambda_consistency
        self.config = config
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # First forward pass (adversarial path according to current intervention settings)
        outputs_adv, _ = super().compute_loss(model, inputs, return_outputs=True)

        # Second forward pass with adversarial intervention disabled to get clean logits
        original_adv_epsilons = {}
        if hasattr(model, 'interventions'):
            for module_key, intervention in model.interventions.items():
                if hasattr(intervention, 'adv_epsilon'):
                    original_adv_epsilons[module_key] = float(intervention.adv_epsilon)
                    intervention.adv_epsilon = 0.0

        try:
            with torch.no_grad():
                outputs_clean, _ = super().compute_loss(model, inputs, return_outputs=True)
        finally:
            # Restore adversarial epsilons
            for module_key, epsilon_value in original_adv_epsilons.items():
                model.interventions[module_key].adv_epsilon = epsilon_value

        # Base loss from adversarial forward
        total_loss = outputs_adv.loss
        # Compute KL divergence at final layer (logits) between clean and adversarial outputs
        if self.lambda_consistency > 0:
            logits_adv = outputs_adv.logits  # [B, T, V]
            logits_clean = outputs_clean.logits  # [B, T, V]

            log_p_adv = F.log_softmax(logits_adv, dim=-1)
            p_clean = F.softmax(logits_clean, dim=-1)
            # token-wise KL(p_clean || p_adv)
            kl_per_token = F.kl_div(log_p_adv, p_clean, reduction='none').sum(dim=-1)  # [B, T]

            # Masking if attention_mask provided
            if 'attention_mask' in inputs and inputs['attention_mask'] is not None:
                mask = inputs['attention_mask'].to(kl_per_token.dtype)
                kl_sum = (kl_per_token * mask).sum()
                denom = mask.sum().clamp_min(1.0)
                kl_loss = kl_sum / denom
            else:
                kl_loss = kl_per_token.mean()
            total_loss = total_loss + self.lambda_consistency * kl_loss

        return (total_loss, outputs_adv) if return_outputs else total_loss


class ReftTrainerCertified(ReftTrainerForCausalLM):
    """
    ReftTrainer with Spectral regularization for controlling global Lipschitz constant.
    Follows the same pattern as ReftTrainerAdv but adds spectral regularization based on A = I + R⊤(W - R).
    """
    def __init__(self, lambda_consistency=0.1, config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_spectral = lambda_consistency
        self.config = config
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # First forward pass (standard forward pass)
        outputs, _ = super().compute_loss(model, inputs, return_outputs=True)
        
        # Base loss from standard forward pass
        total_loss = outputs.loss
        
        
        if self.lambda_spectral > 0:
            spectral_penalty = self._compute_spectral_penalty(model)
            total_loss = total_loss + self.lambda_spectral * spectral_penalty
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _compute_spectral_penalty(self, model,accelerate=True):
        """
        Compute spectral penalty based on A = I + R⊤(W - R).
        For LoReFT interventions: A = I + R⊤(W - R) where R is rotate_layer.weight and W is learned_source.weight.
        """
        spectral_penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        
        if hasattr(model, 'interventions'):
            for module_key, intervention in model.interventions.items():
                if (hasattr(intervention, 'learned_source') and 
                    hasattr(intervention, 'rotate_layer') and
                    hasattr(intervention.learned_source, 'weight') and
                    hasattr(intervention.rotate_layer, 'weight')):
                    
                    # Get W (learned_source weight) and R (rotate_layer weight)
                    W = intervention.learned_source.weight  # Shape: [low_rank_dim, embed_dim]
                    R = intervention.rotate_layer.weight    # Shape: [embed_dim, low_rank_dim]
                    
                    # Ensure compatible dimensions for matrix operations
                    if W.shape[1] == R.shape[0]:  # embed_dim should match
                        # Compute A = I + R⊤(W - R)
                        # R⊤ has shape [low_rank_dim, embed_dim]
                        # (W - R) needs proper dimension handling
                        
                        # Method 1: If W and R⊤ have same dimensions
                       
                        if accelerate is False:
                            if W.shape == R.T.shape:
                                I = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
                                A = I + R @ (W - R.T)
                                sigma_max = torch.linalg.svdvals(A).max()
                                spectral_penalty += sigma_max
                        else:
                            
                            R=R.to(torch.bfloat16)
                            W=W.to(torch.bfloat16)
                            loss=torch.norm(R@(W-R.T),p=2)+1
                            spectral_penalty += loss
                      
        return spectral_penalty