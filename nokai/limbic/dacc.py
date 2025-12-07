"""
dACC - Dorsal Anterior Cingulate Cortex
       The Metacognitive Monitor

Biological Parallel:
    The dACC is the brain's "conflict detector" and uncertainty
    monitor. It activates when:
    - Multiple response options compete
    - Errors are detected
    - Expected outcomes don't match reality
    - Cognitive effort is needed
    
    Functions:
    1. Error detection and signaling
    2. Conflict monitoring
    3. Uncertainty estimation
    4. Cognitive control allocation

Implementation:
    We measure confidence/uncertainty across the system and 
    determine when to "think more carefully" (allocate more 
    compute) vs. give a quick response.
    
Efficiency:
    - O(1) uncertainty estimation
    - Sparse activation for routine decisions
    - Energy-aware compute allocation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum
import math


class CognitiveState(Enum):
    """Cognitive control states based on dACC monitoring."""
    AUTOMATIC = "automatic"       # Low conflict, proceed quickly
    MONITORING = "monitoring"     # Some uncertainty, careful processing
    DELIBERATE = "deliberate"     # High conflict, slow, careful thinking
    ERROR_RECOVERY = "error"      # Error detected, reconsider


@dataclass
class MetacognitiveAssessment:
    """
    Assessment of current cognitive state.
    
    Biological Mapping:
        - confidence: How certain are we? (inversely related to dACC activation)
        - conflict: Are there competing responses? (response conflict)
        - error_likelihood: Probability of error (error-related negativity)
        - cognitive_load: Current processing demands
        - recommended_state: Suggested cognitive control mode
    """
    confidence: float
    conflict: float
    error_likelihood: float
    cognitive_load: float
    recommended_state: CognitiveState


class MetacognitiveMonitor(nn.Module):
    """
    dACC - The Brain's Quality Controller
    
    Biological Parallel:
        The dorsal anterior cingulate cortex monitors cognitive
        performance and signals when more careful processing is needed.
        
        Key signals:
        - Error-Related Negativity (ERN): Fires when error is made
        - Conflict Signal: Fires when responses compete
        - Uncertainty: Activates prefrontal for deliberation
        
    Implementation:
        We track multiple sources of uncertainty/conflict and
        integrate them into a unified "need for control" signal.
        This gates whether to:
        1. Respond quickly (automatic mode)
        2. Allocate more computation (deliberate mode)
        3. Hold response and reconsider (error recovery)
    
    Efficiency:
        - Minimal overhead in "automatic" mode
        - Scales computation based on need
        - Energy-conscious processing
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_response_options: int = 16,
        confidence_threshold: float = 0.7,
        conflict_threshold: float = 0.3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.confidence_threshold = confidence_threshold
        self.conflict_threshold = conflict_threshold
        
        # ============================================
        # CONFIDENCE ESTIMATOR
        # ============================================
        # How certain is the system about its current output?
        self.confidence_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # ============================================
        # CONFLICT DETECTOR
        # ============================================
        # Are multiple responses competing?
        self.conflict_net = nn.Sequential(
            nn.Linear(state_dim + num_response_options, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # ============================================
        # ERROR PREDICTOR
        # ============================================
        # Predict likelihood of error based on current state
        self.error_predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # ============================================
        # COGNITIVE LOAD ESTIMATOR
        # ============================================
        self.load_estimator = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        
        # History tracking for error patterns
        self.register_buffer('recent_errors', torch.zeros(100))
        self.register_buffer('error_ptr', torch.tensor(0, dtype=torch.long))
        
        # Adaptive thresholds (learned from experience)
        self.adaptive_confidence_threshold = nn.Parameter(
            torch.tensor(confidence_threshold)
        )
        self.adaptive_conflict_threshold = nn.Parameter(
            torch.tensor(conflict_threshold)
        )
        
        # Synaptic weights for metacognitive learning
        self.register_buffer('false_positive_rate', torch.tensor(0.0))
        self.register_buffer('false_negative_rate', torch.tensor(0.0))
    
    def forward(
        self,
        state: torch.Tensor,
        response_distribution: Optional[torch.Tensor] = None,
        previous_error: Optional[bool] = None,
    ) -> Tuple[MetacognitiveAssessment, Dict]:
        """
        Assess current cognitive state and determine control needs.
        
        Args:
            state: Current internal state [batch, state_dim]
            response_distribution: Distribution over possible responses
            previous_error: Whether an error occurred on last trial
            
        Returns:
            assessment: Metacognitive assessment
            metadata: Detailed signals for debugging
        """
        batch_size = state.shape[0]
        
        # ============================================
        # COMPUTE CONFIDENCE
        # ============================================
        confidence = self.confidence_net(state)
        
        # ============================================
        # COMPUTE CONFLICT (if response distribution available)
        # ============================================
        if response_distribution is not None:
            # Conflict is high when response distribution is flat (uncertain)
            # Entropy-based conflict measure
            if response_distribution.dim() == 1:
                response_distribution = response_distribution.unsqueeze(0)
            
            # Pad or truncate to expected size
            combined = torch.cat([
                state, 
                F.pad(response_distribution, (0, max(0, 16 - response_distribution.shape[-1])))[:, :16]
            ], dim=-1)
            conflict = self.conflict_net(combined)
        else:
            # Default conflict based on state alone
            conflict = torch.zeros(batch_size, 1, device=state.device)
        
        # ============================================
        # ESTIMATE ERROR LIKELIHOOD
        # ============================================
        error_likelihood = self.error_predictor(state)
        
        # Adjust based on recent error history
        recent_error_rate = self.recent_errors.mean()
        error_likelihood = error_likelihood * (1 + 0.5 * recent_error_rate)
        
        # ============================================
        # ESTIMATE COGNITIVE LOAD
        # ============================================
        cognitive_load = self.load_estimator(state)
        
        # ============================================
        # DETERMINE COGNITIVE STATE
        # ============================================
        conf = confidence.mean().item()
        conf_threshold = torch.sigmoid(self.adaptive_confidence_threshold).item()
        conf_threshold = self.confidence_threshold  # Use fixed for now
        
        confl = conflict.mean().item()
        conf_thresh = self.conflict_threshold
        
        err = error_likelihood.mean().item()
        
        # Decision tree for cognitive state
        if previous_error:
            recommended_state = CognitiveState.ERROR_RECOVERY
        elif conf < conf_threshold * 0.5 or confl > 0.7 or err > 0.5:
            recommended_state = CognitiveState.DELIBERATE
        elif conf < conf_threshold or confl > conf_thresh:
            recommended_state = CognitiveState.MONITORING
        else:
            recommended_state = CognitiveState.AUTOMATIC
        
        # Update error history
        if previous_error is not None:
            ptr = self.error_ptr.item()
            self.recent_errors[ptr] = 1.0 if previous_error else 0.0
            self.error_ptr = (self.error_ptr + 1) % 100
        
        # Create assessment
        assessment = MetacognitiveAssessment(
            confidence=conf,
            conflict=confl,
            error_likelihood=err,
            cognitive_load=cognitive_load.mean().item(),
            recommended_state=recommended_state,
        )
        
        metadata = {
            'confidence': conf,
            'conflict': confl,
            'error_likelihood': err,
            'cognitive_load': cognitive_load.mean().item(),
            'state': recommended_state.value,
            'recent_error_rate': recent_error_rate.item(),
            'should_deliberate': recommended_state in [
                CognitiveState.DELIBERATE, 
                CognitiveState.ERROR_RECOVERY
            ],
        }
        
        return assessment, metadata
    
    def get_compute_allocation(
        self, 
        assessment: MetacognitiveAssessment,
    ) -> Dict[str, float]:
        """
        Determine compute resource allocation based on assessment.
        
        Returns:
            Dict with allocation multipliers for different subsystems
        """
        if assessment.recommended_state == CognitiveState.AUTOMATIC:
            return {
                'attention_heads': 0.5,
                'working_memory': 0.5,
                'deliberation_iterations': 1,
                'memory_retrieval_k': 3,
            }
        elif assessment.recommended_state == CognitiveState.MONITORING:
            return {
                'attention_heads': 0.75,
                'working_memory': 0.75,
                'deliberation_iterations': 2,
                'memory_retrieval_k': 5,
            }
        elif assessment.recommended_state == CognitiveState.DELIBERATE:
            return {
                'attention_heads': 1.0,
                'working_memory': 1.0,
                'deliberation_iterations': 4,
                'memory_retrieval_k': 10,
            }
        else:  # ERROR_RECOVERY
            return {
                'attention_heads': 1.0,
                'working_memory': 1.0,
                'deliberation_iterations': 6,
                'memory_retrieval_k': 15,
            }
    
    def energy_check(self, state: torch.Tensor) -> bool:
        """
        Quick check if detailed metacognitive assessment is needed.
        
        Most routine processing doesn't need full metacognitive monitoring.
        Only activate when state is unusual or stakes are high.
        """
        # Simple heuristic: high variance in state suggests uncertainty
        state_variance = state.var(dim=-1).mean().item()
        return state_variance > 0.5
    
    def record_outcome(
        self,
        predicted_state: CognitiveState,
        actual_error: bool,
    ):
        """
        Record outcome for calibration.
        
        Biological Parallel:
            The dACC learns to better predict errors over time.
            This is similar to reinforcement of error detection circuits.
        """
        if predicted_state == CognitiveState.AUTOMATIC and actual_error:
            # Should have been more careful (false negative)
            self.false_negative_rate = 0.9 * self.false_negative_rate + 0.1
        elif predicted_state in [CognitiveState.DELIBERATE, CognitiveState.ERROR_RECOVERY] and not actual_error:
            # Was too cautious (false positive)
            self.false_positive_rate = 0.9 * self.false_positive_rate + 0.1
        else:
            # Correct calibration
            self.false_positive_rate *= 0.95
            self.false_negative_rate *= 0.95


class ConflictSignal(nn.Module):
    """
    Measures response conflict from multiple competing activations.
    
    Biological Parallel:
        When the brain activates multiple incompatible response 
        options, the dACC detects this conflict and signals for
        top-down control to resolve it.
    """
    
    def __init__(self, num_responses: int):
        super().__init__()
        self.num_responses = num_responses
    
    def forward(self, response_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute conflict from response logits.
        
        High conflict = multiple responses have similar activation.
        Low conflict = one response dominates.
        
        Args:
            response_logits: Raw response activations [batch, num_responses]
            
        Returns:
            conflict: Conflict score [batch]
        """
        # Convert to probabilities
        probs = F.softmax(response_logits, dim=-1)
        
        # Entropy-based conflict
        # Max entropy = max conflict (uniform distribution)
        # Min entropy = min conflict (one response dominates)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        max_entropy = math.log(self.num_responses)
        
        conflict = entropy / max_entropy  # Normalize to [0, 1]
        
        return conflict
    
    def from_top_k(self, response_logits: torch.Tensor, k: int = 2) -> torch.Tensor:
        """
        Alternative: Conflict from ratio of top-k responses.
        
        High conflict when top-2 responses are close in activation.
        """
        top_k, _ = torch.topk(response_logits, k, dim=-1)
        
        # Ratio of second-best to best
        conflict = top_k[:, 1] / (top_k[:, 0] + 1e-10)
        
        return conflict.clamp(0, 1)
