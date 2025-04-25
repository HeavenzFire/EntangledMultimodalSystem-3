from typing import Dict, Any, List
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ValidationStatus(Enum):
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    status: ValidationStatus
    message: str
    details: Dict[str, Any]
    timestamp: float

class QuantumValidator:
    def __init__(self):
        self.validation_history: List[ValidationResult] = []
        self.thresholds = {
            'security': {
                'key_strength': 256,
                'entropy': 0.8,
                'coherence': 0.9
            },
            'synthesis': {
                'state_fidelity': 0.95,
                'resonance': 0.85,
                'harmony': 0.9
            },
            'torus': {
                'field_stability': 0.8,
                'alignment': 0.85,
                'coherence': 0.9
            }
        }

    def validate_security(self, metrics: Dict[str, Any]) -> ValidationResult:
        """Validate security metrics"""
        status = ValidationStatus.VALID
        messages = []
        details = {}

        # Check key strength
        if metrics.get('key_strength', 0) < self.thresholds['security']['key_strength']:
            status = ValidationStatus.ERROR
            messages.append("Key strength below threshold")
            details['key_strength'] = metrics.get('key_strength')

        # Check entropy
        if metrics.get('entropy', 0) < self.thresholds['security']['entropy']:
            status = ValidationStatus.WARNING
            messages.append("Entropy level below optimal")
            details['entropy'] = metrics.get('entropy')

        # Check coherence
        if metrics.get('coherence', 0) < self.thresholds['security']['coherence']:
            status = ValidationStatus.CRITICAL
            messages.append("Security coherence compromised")
            details['coherence'] = metrics.get('coherence')

        return ValidationResult(
            status=status,
            message=", ".join(messages) if messages else "Security validation passed",
            details=details,
            timestamp=np.datetime64('now').astype(float)
        )

    def validate_synthesis(self, metrics: Dict[str, Any]) -> ValidationResult:
        """Validate synthesis metrics"""
        status = ValidationStatus.VALID
        messages = []
        details = {}

        # Check state fidelity
        if metrics.get('state_fidelity', 0) < self.thresholds['synthesis']['state_fidelity']:
            status = ValidationStatus.ERROR
            messages.append("State fidelity below threshold")
            details['state_fidelity'] = metrics.get('state_fidelity')

        # Check resonance
        if metrics.get('resonance', 0) < self.thresholds['synthesis']['resonance']:
            status = ValidationStatus.WARNING
            messages.append("Resonance level below optimal")
            details['resonance'] = metrics.get('resonance')

        # Check harmony
        if metrics.get('harmony', 0) < self.thresholds['synthesis']['harmony']:
            status = ValidationStatus.CRITICAL
            messages.append("Synthesis harmony compromised")
            details['harmony'] = metrics.get('harmony')

        return ValidationResult(
            status=status,
            message=", ".join(messages) if messages else "Synthesis validation passed",
            details=details,
            timestamp=np.datetime64('now').astype(float)
        )

    def validate_torus(self, metrics: Dict[str, Any]) -> ValidationResult:
        """Validate torus metrics"""
        status = ValidationStatus.VALID
        messages = []
        details = {}

        # Check field stability
        if metrics.get('field_stability', 0) < self.thresholds['torus']['field_stability']:
            status = ValidationStatus.ERROR
            messages.append("Field stability below threshold")
            details['field_stability'] = metrics.get('field_stability')

        # Check alignment
        if metrics.get('alignment', 0) < self.thresholds['torus']['alignment']:
            status = ValidationStatus.WARNING
            messages.append("Alignment level below optimal")
            details['alignment'] = metrics.get('alignment')

        # Check coherence
        if metrics.get('coherence', 0) < self.thresholds['torus']['coherence']:
            status = ValidationStatus.CRITICAL
            messages.append("Torus coherence compromised")
            details['coherence'] = metrics.get('coherence')

        return ValidationResult(
            status=status,
            message=", ".join(messages) if messages else "Torus validation passed",
            details=details,
            timestamp=np.datetime64('now').astype(float)
        )

    def validate_system(self, metrics: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """Validate entire system"""
        results = {
            'security': self.validate_security(metrics.get('security', {})),
            'synthesis': self.validate_synthesis(metrics.get('synthesis', {})),
            'torus': self.validate_torus(metrics.get('torus', {}))
        }
        
        # Add to validation history
        self.validation_history.extend(results.values())
        
        # Keep only last 1000 validations
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
        
        return results

    def get_validation_history(self, limit: int = 100) -> List[ValidationResult]:
        """Get recent validation history"""
        return self.validation_history[-limit:]

    def get_system_status(self) -> ValidationStatus:
        """Get overall system status"""
        if not self.validation_history:
            return ValidationStatus.VALID
        
        # Get the most recent validation results
        recent_results = self.get_validation_history(10)
        
        # Determine overall status based on most critical status
        status_priority = {
            ValidationStatus.CRITICAL: 4,
            ValidationStatus.ERROR: 3,
            ValidationStatus.WARNING: 2,
            ValidationStatus.VALID: 1
        }
        
        return max(recent_results, key=lambda x: status_priority[x.status]).status 