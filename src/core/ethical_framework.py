import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Optional
from src.utils.errors import ModelError
from src.utils.logger import logger

class EthicalFramework:
    """Ethical Framework for moral reasoning and decision-making."""
    
    def __init__(self):
        """Initialize the ethical framework."""
        try:
            # Initialize ethical parameters
            self.params = {
                "utilitarian_weight": 0.4,
                "deontological_weight": 0.3,
                "virtue_weight": 0.3,
                "fairness_threshold": 0.7,
                "harm_threshold": 0.3
            }
            
            # Initialize ethical models
            self.models = {
                "utilitarian": self._build_utilitarian_model(),
                "deontological": self._build_deontological_model(),
                "virtue": self._build_virtue_model()
            }
            
            # Initialize ethical state
            self.ethics = {
                "moral_values": np.zeros(6),  # [fairness, care, loyalty, authority, sanctity, liberty]
                "ethical_principles": {},
                "moral_dilemmas": [],
                "ethical_decisions": []
            }
            
            # Initialize performance metrics
            self.metrics = {
                "ethical_alignment": 0.0,
                "moral_consistency": 0.0,
                "fairness_score": 0.0,
                "harm_score": 0.0
            }
            
            logger.info("EthicalFramework initialized")
            
        except Exception as e:
            logger.error(f"Error initializing EthicalFramework: {str(e)}")
            raise ModelError(f"Failed to initialize EthicalFramework: {str(e)}")

    def evaluate_action(self, action: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate an action from multiple ethical perspectives."""
        try:
            # Evaluate from utilitarian perspective
            utilitarian_score = self._evaluate_utilitarian(action)
            
            # Evaluate from deontological perspective
            deontological_score = self._evaluate_deontological(action)
            
            # Evaluate from virtue perspective
            virtue_score = self._evaluate_virtue(action)
            
            # Calculate weighted ethical score
            ethical_score = (
                utilitarian_score * self.params["utilitarian_weight"] +
                deontological_score * self.params["deontological_weight"] +
                virtue_score * self.params["virtue_weight"]
            )
            
            # Check fairness and harm
            fairness = self._check_fairness(action)
            harm = self._check_harm(action)
            
            return {
                "ethical_score": ethical_score,
                "utilitarian_score": utilitarian_score,
                "deontological_score": deontological_score,
                "virtue_score": virtue_score,
                "fairness": fairness,
                "harm": harm
            }
            
        except Exception as e:
            logger.error(f"Error evaluating action: {str(e)}")
            raise ModelError(f"Action evaluation failed: {str(e)}")

    def resolve_dilemma(self, dilemma: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve an ethical dilemma."""
        try:
            # Analyze dilemma
            analysis = self._analyze_dilemma(dilemma)
            
            # Evaluate options
            options = self._evaluate_options(dilemma["options"])
            
            # Apply ethical principles
            principles = self._apply_principles(analysis, options)
            
            # Make decision
            decision = self._make_decision(principles)
            
            # Update ethical state
            self._update_ethical_state(dilemma, decision)
            
            return {
                "analysis": analysis,
                "options": options,
                "principles": principles,
                "decision": decision
            }
            
        except Exception as e:
            logger.error(f"Error resolving dilemma: {str(e)}")
            raise ModelError(f"Dilemma resolution failed: {str(e)}")

    def get_state(self) -> Dict[str, Any]:
        """Get current ethical state."""
        return {
            "ethics": self.ethics,
            "metrics": self.metrics
        }

    def reset(self) -> None:
        """Reset ethical framework to initial state."""
        try:
            # Reset ethical state
            self.ethics.update({
                "moral_values": np.zeros(6),
                "ethical_principles": {},
                "moral_dilemmas": [],
                "ethical_decisions": []
            })
            
            # Reset metrics
            self.metrics.update({
                "ethical_alignment": 0.0,
                "moral_consistency": 0.0,
                "fairness_score": 0.0,
                "harm_score": 0.0
            })
            
            logger.info("EthicalFramework reset completed")
            
        except Exception as e:
            logger.error(f"Error resetting EthicalFramework: {str(e)}")
            raise ModelError(f"EthicalFramework reset failed: {str(e)}")

    def _build_utilitarian_model(self) -> tf.keras.Model:
        """Build utilitarian ethical model."""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building utilitarian model: {str(e)}")
            raise ModelError(f"Utilitarian model building failed: {str(e)}")

    def _build_deontological_model(self) -> tf.keras.Model:
        """Build deontological ethical model."""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building deontological model: {str(e)}")
            raise ModelError(f"Deontological model building failed: {str(e)}")

    def _build_virtue_model(self) -> tf.keras.Model:
        """Build virtue ethical model."""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building virtue model: {str(e)}")
            raise ModelError(f"Virtue model building failed: {str(e)}")

    def _evaluate_utilitarian(self, action: Dict[str, Any]) -> float:
        """Evaluate action from utilitarian perspective."""
        try:
            # Extract consequences
            consequences = np.array([
                action.get("happiness", 0.0),
                action.get("suffering", 0.0),
                action.get("utility", 0.0),
                action.get("efficiency", 0.0),
                action.get("benefit", 0.0),
                action.get("cost", 0.0)
            ])
            
            # Normalize consequences
            normalized = (consequences - np.min(consequences)) / (np.max(consequences) - np.min(consequences))
            
            # Get model prediction
            score = self.models["utilitarian"].predict(normalized.reshape(1, -1))[0][0]
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating utilitarian: {str(e)}")
            raise ModelError(f"Utilitarian evaluation failed: {str(e)}")

    def _evaluate_deontological(self, action: Dict[str, Any]) -> float:
        """Evaluate action from deontological perspective."""
        try:
            # Extract principles
            principles = np.array([
                action.get("duty", 0.0),
                action.get("rights", 0.0),
                action.get("justice", 0.0),
                action.get("autonomy", 0.0),
                action.get("respect", 0.0),
                action.get("obligation", 0.0)
            ])
            
            # Normalize principles
            normalized = (principles - np.min(principles)) / (np.max(principles) - np.min(principles))
            
            # Get model prediction
            score = self.models["deontological"].predict(normalized.reshape(1, -1))[0][0]
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating deontological: {str(e)}")
            raise ModelError(f"Deontological evaluation failed: {str(e)}")

    def _evaluate_virtue(self, action: Dict[str, Any]) -> float:
        """Evaluate action from virtue perspective."""
        try:
            # Extract virtues
            virtues = np.array([
                action.get("courage", 0.0),
                action.get("wisdom", 0.0),
                action.get("temperance", 0.0),
                action.get("justice", 0.0),
                action.get("compassion", 0.0),
                action.get("integrity", 0.0)
            ])
            
            # Normalize virtues
            normalized = (virtues - np.min(virtues)) / (np.max(virtues) - np.min(virtues))
            
            # Get model prediction
            score = self.models["virtue"].predict(normalized.reshape(1, -1))[0][0]
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating virtue: {str(e)}")
            raise ModelError(f"Virtue evaluation failed: {str(e)}")

    def _check_fairness(self, action: Dict[str, Any]) -> float:
        """Check fairness of action."""
        try:
            # Extract fairness metrics
            fairness = np.array([
                action.get("equality", 0.0),
                action.get("impartiality", 0.0),
                action.get("transparency", 0.0),
                action.get("accountability", 0.0),
                action.get("inclusivity", 0.0),
                action.get("diversity", 0.0)
            ])
            
            # Calculate fairness score
            score = np.mean(fairness)
            
            return score
            
        except Exception as e:
            logger.error(f"Error checking fairness: {str(e)}")
            raise ModelError(f"Fairness check failed: {str(e)}")

    def _check_harm(self, action: Dict[str, Any]) -> float:
        """Check potential harm of action."""
        try:
            # Extract harm metrics
            harm = np.array([
                action.get("physical_harm", 0.0),
                action.get("psychological_harm", 0.0),
                action.get("social_harm", 0.0),
                action.get("environmental_harm", 0.0),
                action.get("economic_harm", 0.0),
                action.get("cultural_harm", 0.0)
            ])
            
            # Calculate harm score
            score = np.mean(harm)
            
            return score
            
        except Exception as e:
            logger.error(f"Error checking harm: {str(e)}")
            raise ModelError(f"Harm check failed: {str(e)}")

    def _analyze_dilemma(self, dilemma: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ethical dilemma."""
        try:
            # Extract dilemma components
            analysis = {
                "stakeholders": dilemma.get("stakeholders", []),
                "values": dilemma.get("values", []),
                "conflicts": dilemma.get("conflicts", []),
                "context": dilemma.get("context", {})
            }
            
            # Calculate moral values
            analysis["moral_values"] = self._calculate_moral_values(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing dilemma: {str(e)}")
            raise ModelError(f"Dilemma analysis failed: {str(e)}")

    def _evaluate_options(self, options: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Evaluate possible options."""
        try:
            evaluated = []
            for option in options:
                # Evaluate each option
                evaluation = self.evaluate_action(option)
                evaluated.append(evaluation)
            
            return evaluated
            
        except Exception as e:
            logger.error(f"Error evaluating options: {str(e)}")
            raise ModelError(f"Option evaluation failed: {str(e)}")

    def _apply_principles(self, analysis: Dict[str, Any], options: List[Dict[str, float]]) -> Dict[str, Any]:
        """Apply ethical principles to options."""
        try:
            principles = {
                "utilitarian": [],
                "deontological": [],
                "virtue": [],
                "fairness": [],
                "harm": []
            }
            
            for option in options:
                # Apply each principle
                principles["utilitarian"].append(option["utilitarian_score"])
                principles["deontological"].append(option["deontological_score"])
                principles["virtue"].append(option["virtue_score"])
                principles["fairness"].append(option["fairness"])
                principles["harm"].append(option["harm"])
            
            return principles
            
        except Exception as e:
            logger.error(f"Error applying principles: {str(e)}")
            raise ModelError(f"Principle application failed: {str(e)}")

    def _make_decision(self, principles: Dict[str, List[float]]) -> Dict[str, Any]:
        """Make ethical decision based on principles."""
        try:
            # Calculate weighted scores
            scores = []
            for i in range(len(principles["utilitarian"])):
                score = (
                    principles["utilitarian"][i] * self.params["utilitarian_weight"] +
                    principles["deontological"][i] * self.params["deontological_weight"] +
                    principles["virtue"][i] * self.params["virtue_weight"]
                )
                scores.append(score)
            
            # Select best option
            best_idx = np.argmax(scores)
            
            return {
                "option": best_idx,
                "score": scores[best_idx],
                "principles": {
                    "utilitarian": principles["utilitarian"][best_idx],
                    "deontological": principles["deontological"][best_idx],
                    "virtue": principles["virtue"][best_idx],
                    "fairness": principles["fairness"][best_idx],
                    "harm": principles["harm"][best_idx]
                }
            }
            
        except Exception as e:
            logger.error(f"Error making decision: {str(e)}")
            raise ModelError(f"Decision making failed: {str(e)}")

    def _update_ethical_state(self, dilemma: Dict[str, Any], decision: Dict[str, Any]) -> None:
        """Update ethical state with new dilemma and decision."""
        try:
            # Add to moral dilemmas
            self.ethics["moral_dilemmas"].append(dilemma)
            
            # Add to ethical decisions
            self.ethics["ethical_decisions"].append(decision)
            
            # Update moral values
            self._update_moral_values(dilemma, decision)
            
            # Update metrics
            self._update_metrics()
            
        except Exception as e:
            logger.error(f"Error updating ethical state: {str(e)}")
            raise ModelError(f"Ethical state update failed: {str(e)}")

    def _calculate_moral_values(self, analysis: Dict[str, Any]) -> np.ndarray:
        """Calculate moral values from analysis."""
        try:
            # Initialize values
            values = np.zeros(6)
            
            # Calculate from stakeholders
            for stakeholder in analysis["stakeholders"]:
                values += stakeholder.get("values", np.zeros(6))
            
            # Calculate from conflicts
            for conflict in analysis["conflicts"]:
                values += conflict.get("values", np.zeros(6))
            
            # Normalize values
            values = values / np.sum(np.abs(values))
            
            return values
            
        except Exception as e:
            logger.error(f"Error calculating moral values: {str(e)}")
            raise ModelError(f"Moral value calculation failed: {str(e)}")

    def _update_moral_values(self, dilemma: Dict[str, Any], decision: Dict[str, Any]) -> None:
        """Update moral values based on dilemma and decision."""
        try:
            # Get current values
            current = self.ethics["moral_values"]
            
            # Calculate new values
            new = self._calculate_moral_values({
                "stakeholders": dilemma.get("stakeholders", []),
                "values": dilemma.get("values", []),
                "conflicts": dilemma.get("conflicts", []),
                "context": dilemma.get("context", {})
            })
            
            # Update with learning rate
            learning_rate = 0.1
            self.ethics["moral_values"] = (1 - learning_rate) * current + learning_rate * new
            
        except Exception as e:
            logger.error(f"Error updating moral values: {str(e)}")
            raise ModelError(f"Moral value update failed: {str(e)}")

    def _update_metrics(self) -> None:
        """Update performance metrics."""
        try:
            # Calculate ethical alignment
            self.metrics["ethical_alignment"] = self._calculate_ethical_alignment()
            
            # Calculate moral consistency
            self.metrics["moral_consistency"] = self._calculate_moral_consistency()
            
            # Calculate fairness score
            self.metrics["fairness_score"] = self._calculate_fairness_score()
            
            # Calculate harm score
            self.metrics["harm_score"] = self._calculate_harm_score()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            raise ModelError(f"Metrics update failed: {str(e)}")

    def _calculate_ethical_alignment(self) -> float:
        """Calculate ethical alignment score."""
        try:
            # Get recent decisions
            decisions = self.ethics["ethical_decisions"][-10:]
            
            if not decisions:
                return 0.0
            
            # Calculate alignment
            alignment = np.mean([
                decision["score"] for decision in decisions
            ])
            
            return alignment
            
        except Exception as e:
            logger.error(f"Error calculating ethical alignment: {str(e)}")
            raise ModelError(f"Ethical alignment calculation failed: {str(e)}")

    def _calculate_moral_consistency(self) -> float:
        """Calculate moral consistency score."""
        try:
            # Get recent decisions
            decisions = self.ethics["ethical_decisions"][-10:]
            
            if not decisions:
                return 0.0
            
            # Calculate consistency
            scores = np.array([
                decision["score"] for decision in decisions
            ])
            consistency = 1.0 - np.std(scores)
            
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating moral consistency: {str(e)}")
            raise ModelError(f"Moral consistency calculation failed: {str(e)}")

    def _calculate_fairness_score(self) -> float:
        """Calculate overall fairness score."""
        try:
            # Get recent decisions
            decisions = self.ethics["ethical_decisions"][-10:]
            
            if not decisions:
                return 0.0
            
            # Calculate fairness
            fairness = np.mean([
                decision["principles"]["fairness"] for decision in decisions
            ])
            
            return fairness
            
        except Exception as e:
            logger.error(f"Error calculating fairness score: {str(e)}")
            raise ModelError(f"Fairness score calculation failed: {str(e)}")

    def _calculate_harm_score(self) -> float:
        """Calculate overall harm score."""
        try:
            # Get recent decisions
            decisions = self.ethics["ethical_decisions"][-10:]
            
            if not decisions:
                return 0.0
            
            # Calculate harm
            harm = np.mean([
                decision["principles"]["harm"] for decision in decisions
            ])
            
            return harm
            
        except Exception as e:
            logger.error(f"Error calculating harm score: {str(e)}")
            raise ModelError(f"Harm score calculation failed: {str(e)}") 