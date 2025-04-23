import unittest
import numpy as np
import yaml
from core.medical_doctor import AGIMedicalDoctor
from pathlib import Path
from datetime import datetime, timedelta

class TestAGIMedicalDoctor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load configuration
        config_path = Path("config/config.yaml")
        with open(config_path, 'r') as f:
            cls.config = yaml.safe_load(f)
        
        # Initialize AGI medical doctor
        cls.doctor = AGIMedicalDoctor(
            diagnostic_model_path=cls.config['diagnostic_model_path'],
            tokenizer_path=cls.config['tokenizer_path'],
            interaction_model_path=cls.config['interaction_model_path']
        )

    def test_initialization(self):
        """Test initialization of AGI medical doctor"""
        self.assertIsNotNone(self.doctor.diagnostic_model)
        self.assertIsNotNone(self.doctor.tokenizer)
        self.assertIsNotNone(self.doctor.interaction_model)
        self.assertIsNotNone(self.doctor.quantum_circuit)
        self.assertEqual(self.doctor.quantum_circuit.num_qubits, 
                        self.config['quantum_circuit']['num_qubits'])
        
        # Verify ethical compliance initialization
        self.assertIsNotNone(self.doctor.bias_mitigation_model)
        self.assertIsNotNone(self.doctor.explainability_engine)
        self.assertIsNotNone(self.doctor.audit_logger)

    def test_genomic_analysis(self):
        """Test genomic data analysis"""
        # Create sample genomic data
        genomic_data = {
            'variants': np.random.rand(100, 4),
            'confidence': np.random.rand(100)
        }
        
        analysis = self.doctor._analyze_genomic_data(genomic_data)
        self.assertIsInstance(analysis, dict)
        self.assertIn('risk_factors', analysis)
        self.assertIn('treatment_implications', analysis)

    def test_biosensor_analysis(self):
        """Test biosensor data analysis"""
        # Create sample biosensor data
        biosensor_data = {
            'heart_rate': np.random.rand(1000),
            'blood_pressure': np.random.rand(1000, 2),
            'temperature': np.random.rand(1000)
        }
        
        analysis = self.doctor._analyze_biosensor_data(biosensor_data)
        self.assertIsInstance(analysis, dict)
        self.assertIn('vital_signs', analysis)
        self.assertIn('anomalies', analysis)

    def test_ehr_analysis(self):
        """Test EHR data analysis"""
        # Create sample EHR data
        ehr_data = {
            'diagnoses': ['hypertension', 'diabetes'],
            'medications': ['metformin', 'lisinopril'],
            'procedures': ['blood_test', 'x-ray'],
            'timestamps': np.arange(10)
        }
        
        analysis = self.doctor._analyze_ehr_data(ehr_data)
        self.assertIsInstance(analysis, dict)
        self.assertIn('medical_history', analysis)
        self.assertIn('treatment_history', analysis)

    def test_behavioral_analysis(self):
        """Test behavioral data analysis"""
        # Create sample behavioral data
        behavioral_data = {
            'activity_levels': np.random.rand(7),
            'sleep_patterns': np.random.rand(7),
            'mood_scores': np.random.rand(7)
        }
        
        analysis = self.doctor._analyze_behavioral_data(behavioral_data)
        self.assertIsInstance(analysis, dict)
        self.assertIn('patterns', analysis)
        self.assertIn('recommendations', analysis)

    def test_multimodal_data_integration(self):
        """Test multimodal data integration capabilities"""
        # Create sample multimodal data
        patient_data = {
            'genomic': {
                'variants': np.random.rand(100, 4),
                'confidence': np.random.rand(100)
            },
            'biosensor': {
                'heart_rate': np.random.rand(1000),
                'blood_pressure': np.random.rand(1000, 2),
                'temperature': np.random.rand(1000),
                'hrv': np.random.rand(1000)  # Heart rate variability
            },
            'environmental': {
                'air_quality': np.random.rand(100),
                'route_safety': np.random.rand(100),
                'pothole_detection': np.random.rand(100)
            },
            'clinical': {
                'ehr': {
                    'diagnoses': ['hypertension', 'diabetes'],
                    'medications': ['metformin', 'lisinopril'],
                    'procedures': ['blood_test', 'x-ray']
                },
                'imaging': np.random.rand(256, 256, 3),  # Sample medical image
                'lab_results': {
                    'cbc': np.random.rand(5),
                    'metabolic_panel': np.random.rand(7)
                }
            }
        }
        
        # Test data integration
        integrated_data = self.doctor._integrate_multimodal_data(patient_data)
        self.assertIsInstance(integrated_data, dict)
        self.assertIn('risk_assessment', integrated_data)
        self.assertIn('treatment_recommendations', integrated_data)
        self.assertIn('environmental_factors', integrated_data)

    def test_advanced_diagnostic_capabilities(self):
        """Test advanced diagnostic capabilities"""
        # Create sample diagnostic data
        diagnostic_data = {
            'symptoms': ['fatigue', 'headache', 'nausea'],
            'vital_signs': {
                'heart_rate': 85,
                'blood_pressure': [120, 80],
                'temperature': 98.6
            },
            'lab_results': {
                'wbc': 7.5,
                'hgb': 14.2,
                'plt': 250
            }
        }
        
        # Test diagnostic process
        diagnosis = self.doctor._perform_advanced_diagnosis(diagnostic_data)
        self.assertIsInstance(diagnosis, dict)
        self.assertIn('primary_diagnosis', diagnosis)
        self.assertIn('differential_diagnoses', diagnosis)
        self.assertIn('confidence_score', diagnosis)
        self.assertIn('explanation', diagnosis)  # SHAP-based explanation

    def test_treatment_optimization(self):
        """Test advanced treatment optimization"""
        # Create sample patient data
        patient_data = {
            'genomic': {'variants': np.random.rand(100, 4)},
            'biosensor': {'heart_rate': np.random.rand(1000)},
            'ehr': {'diagnoses': ['hypertension']},
            'behavioral': {'activity_levels': np.random.rand(7)},
            'environmental': {'air_quality': np.random.rand(100)}
        }
        
        # Test treatment optimization
        treatment_plan = self.doctor.optimize_treatment(patient_data)
        self.assertIsInstance(treatment_plan, dict)
        self.assertIn('medications', treatment_plan)
        self.assertIn('lifestyle_changes', treatment_plan)
        self.assertIn('monitoring_plan', treatment_plan)
        self.assertIn('route_optimization', treatment_plan)  # For cyclists
        self.assertIn('environmental_adaptations', treatment_plan)
        self.assertIn('explanation', treatment_plan)  # SHAP-based explanation

    def test_mental_health_support(self):
        """Test mental health support capabilities"""
        # Create sample mental health data
        mental_health_data = {
            'anxiety_scores': np.random.rand(7),
            'depression_scores': np.random.rand(7),
            'stress_levels': np.random.rand(7),
            'sleep_patterns': np.random.rand(7),
            'activity_levels': np.random.rand(7)
        }
        
        # Test mental health support
        support_plan = self.doctor._generate_mental_health_support(mental_health_data)
        self.assertIsInstance(support_plan, dict)
        self.assertIn('cbsm_modules', support_plan)
        self.assertIn('mindfulness_exercises', support_plan)
        self.assertIn('activity_recommendations', support_plan)
        self.assertIn('crisis_intervention_plan', support_plan)

    def test_ethical_compliance(self):
        """Test ethical compliance and bias mitigation"""
        # Test bias mitigation
        test_cases = [
            {'demographic': 'male', 'age': 45, 'condition': 'pain'},
            {'demographic': 'female', 'age': 45, 'condition': 'pain'},
            {'demographic': 'non-binary', 'age': 45, 'condition': 'pain'}
        ]
        
        recommendations = []
        for case in test_cases:
            rec = self.doctor._generate_treatment_recommendation(case)
            recommendations.append(rec)
        
        # Verify fairness metrics
        fairness_metrics = self.doctor._calculate_fairness_metrics(recommendations)
        self.assertLess(fairness_metrics['demographic_parity'], 0.1)
        self.assertLess(fairness_metrics['equalized_odds'], 0.1)
        
        # Test explainability
        explanation = self.doctor._generate_explanation(recommendations[0])
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)

    def test_performance_metrics(self):
        """Test comprehensive performance metrics"""
        # Update metrics with advanced tracking
        self.doctor.metrics['diagnostic_accuracy'].append(0.95)
        self.doctor.metrics['treatment_success'].append(0.90)
        self.doctor.metrics['patient_satisfaction'].append(0.85)
        self.doctor.metrics['bias_mitigation'].append(0.92)
        self.doctor.metrics['explainability_score'].append(0.88)
        self.doctor.metrics['response_time'].append(6.7)  # seconds
        self.doctor.metrics['hallucination_rate'].append(0.001)
        
        # Verify metrics
        metrics_summary = self.doctor.get_metrics_summary()
        self.assertIsInstance(metrics_summary, dict)
        self.assertIn('average_diagnostic_accuracy', metrics_summary)
        self.assertIn('average_treatment_success', metrics_summary)
        self.assertIn('average_patient_satisfaction', metrics_summary)
        self.assertIn('average_bias_mitigation', metrics_summary)
        self.assertIn('average_explainability_score', metrics_summary)
        self.assertIn('average_response_time', metrics_summary)
        self.assertIn('average_hallucination_rate', metrics_summary)

    def test_urban_cyclist_integration(self):
        """Test urban cyclist-specific healthcare integration"""
        # Create sample cyclist data
        cyclist_data = {
            'strava_data': {
                'route_history': np.random.rand(100, 3),  # lat, long, elevation
                'performance_metrics': {
                    'power_output': np.random.rand(1000),
                    'cadence': np.random.rand(1000),
                    'speed': np.random.rand(1000)
                }
            },
            'environmental_data': {
                'air_quality': {
                    'pm2_5': np.random.rand(100),
                    'no2': np.random.rand(100),
                    'o3': np.random.rand(100)
                },
                'route_safety': {
                    'pothole_density': np.random.rand(100),
                    'traffic_density': np.random.rand(100),
                    'lighting_conditions': np.random.rand(100)
                }
            },
            'health_metrics': {
                'hrv': np.random.rand(1000),  # Heart rate variability
                'respiratory_rate': np.random.rand(1000),
                'oxygen_saturation': np.random.rand(1000)
            }
        }
        
        # Test cyclist-specific analysis
        analysis = self.doctor._analyze_cyclist_data(cyclist_data)
        self.assertIsInstance(analysis, dict)
        self.assertIn('route_recommendations', analysis)
        self.assertIn('health_risk_assessment', analysis)
        self.assertIn('performance_optimization', analysis)
        self.assertIn('environmental_adaptations', analysis)

    def test_cbsm_module_integration(self):
        """Test Cognitive Behavioral Stress Management module integration"""
        # Create sample mental health data
        mental_health_data = {
            'anxiety_scores': {
                'promis_anxiety': np.random.rand(7),
                'generalized_anxiety': np.random.rand(7)
            },
            'stress_levels': {
                'perceived_stress': np.random.rand(7),
                'physiological_stress': np.random.rand(7)
            },
            'coping_mechanisms': {
                'mindfulness_practice': np.random.rand(7),
                'physical_activity': np.random.rand(7),
                'social_support': np.random.rand(7)
            }
        }
        
        # Test CBSM module
        cbsm_plan = self.doctor._generate_cbsm_plan(mental_health_data)
        self.assertIsInstance(cbsm_plan, dict)
        self.assertIn('intervention_modules', cbsm_plan)
        self.assertIn('progress_tracking', cbsm_plan)
        self.assertIn('adaptation_strategy', cbsm_plan)
        
        # Verify expected anxiety reduction
        self.assertLess(cbsm_plan['expected_reduction']['promis_anxiety'], 0.38)

    def test_quantum_enhanced_drug_interactions(self):
        """Test quantum computing-enhanced drug interaction predictions"""
        # Create sample drug interaction data
        drug_data = {
            'current_medications': ['metformin', 'lisinopril', 'aspirin'],
            'proposed_treatment': {
                'drug': 'warfarin',
                'dose': '5mg',
                'frequency': 'daily'
            },
            'patient_characteristics': {
                'age': 45,
                'weight': 70,
                'genetic_markers': np.random.rand(100)
            }
        }
        
        # Test quantum-enhanced prediction
        interaction_analysis = self.doctor._predict_drug_interactions(drug_data)
        self.assertIsInstance(interaction_analysis, dict)
        self.assertIn('interaction_risk', interaction_analysis)
        self.assertIn('recommended_adjustments', interaction_analysis)
        self.assertIn('quantum_certainty_score', interaction_analysis)
        self.assertLess(interaction_analysis['error_rate'], 0.001)

    def test_voice_stress_analysis(self):
        """Test voice stress analysis for mental health assessment"""
        # Create sample voice data
        voice_data = {
            'audio_features': {
                'pitch_variation': np.random.rand(100),
                'speech_rate': np.random.rand(100),
                'voice_tremor': np.random.rand(100)
            },
            'transcript': "I've been feeling more tired than usual and having trouble sleeping",
            'metadata': {
                'duration': 5.2,
                'background_noise': 0.1,
                'recording_quality': 0.95
            }
        }
        
        # Test voice analysis
        analysis = self.doctor._analyze_voice_stress(voice_data)
        self.assertIsInstance(analysis, dict)
        self.assertIn('depression_likelihood', analysis)
        self.assertIn('anxiety_score', analysis)
        self.assertIn('stress_level', analysis)
        self.assertGreater(analysis['detection_accuracy'], 0.89)

    def test_real_time_immunotherapy_monitoring(self):
        """Test real-time immunotherapy monitoring and adjustment"""
        # Create sample immunotherapy data
        immunotherapy_data = {
            'biomarkers': {
                't_cell_count': np.random.rand(100),
                'cytokine_levels': np.random.rand(100, 5),
                'tumor_markers': np.random.rand(100, 3)
            },
            'liquid_biopsy': {
                'circulating_tumor_dna': np.random.rand(100),
                'immune_cell_profile': np.random.rand(100, 4)
            },
            'patient_response': {
                'side_effects': np.random.rand(100),
                'quality_of_life': np.random.rand(100),
                'performance_status': np.random.rand(100)
            }
        }
        
        # Test real-time monitoring
        monitoring_results = self.doctor._monitor_immunotherapy(immunotherapy_data)
        self.assertIsInstance(monitoring_results, dict)
        self.assertIn('dose_adjustment', monitoring_results)
        self.assertIn('response_prediction', monitoring_results)
        self.assertIn('side_effect_risk', monitoring_results)
        self.assertIn('intervention_recommendations', monitoring_results)

    def test_pandemic_resilient_care(self):
        """Test multi-agent reinforcement learning for pandemic-resilient care"""
        # Create sample pandemic scenario data
        scenario_data = {
            'epidemiological_data': {
                'infection_rates': np.random.rand(100),
                'hospital_capacity': np.random.rand(100),
                'vaccination_coverage': np.random.rand(100)
            },
            'patient_population': {
                'risk_factors': np.random.rand(1000, 5),
                'access_to_care': np.random.rand(1000),
                'socioeconomic_status': np.random.rand(1000)
            },
            'healthcare_resources': {
                'staff_availability': np.random.rand(100),
                'equipment_supply': np.random.rand(100),
                'telehealth_capacity': np.random.rand(100)
            }
        }
        
        # Test pandemic response planning
        response_plan = self.doctor._generate_pandemic_response(scenario_data)
        self.assertIsInstance(response_plan, dict)
        self.assertIn('care_delivery_strategy', response_plan)
        self.assertIn('resource_allocation', response_plan)
        self.assertIn('risk_mitigation_measures', response_plan)
        self.assertIn('continuity_of_care_plan', response_plan)

    def test_protein_structure_prediction(self):
        """Test protein structure prediction capabilities"""
        # Create sample protein data
        protein_data = {
            'sequence': 'MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN',
            'target': 'TAAR1',
            'experimental_data': {
                'xray_diffraction': np.random.rand(100, 100),
                'nmr_spectra': np.random.rand(100, 100),
                'cryo_em': np.random.rand(100, 100, 100)
            }
        }
        
        # Test structure prediction
        prediction = self.doctor._predict_protein_structure(protein_data)
        self.assertIsInstance(prediction, dict)
        self.assertIn('3d_structure', prediction)
        self.assertIn('confidence_scores', prediction)
        self.assertIn('binding_sites', prediction)
        self.assertGreater(prediction['accuracy_score'], 0.92)

    def test_generative_molecule_design(self):
        """Test quantum-optimized GAN for molecule design"""
        # Create sample design parameters
        design_params = {
            'target_profile': {
                'binding_affinity': 0.8,
                'selectivity': 0.9,
                'toxicity_threshold': 0.1
            },
            'constraints': {
                'molecular_weight': (200, 500),
                'logp': (-2, 5),
                'rotatable_bonds': (0, 10)
            },
            'optimization_goals': {
                'synthetic_accessibility': 0.8,
                'drug_likeness': 0.9,
                'bioavailability': 0.7
            }
        }
        
        # Test molecule generation
        generated_molecules = self.doctor._generate_molecules(design_params)
        self.assertIsInstance(generated_molecules, list)
        self.assertGreater(len(generated_molecules), 0)
        
        # Verify molecule properties
        for molecule in generated_molecules:
            self.assertIn('smiles', molecule)
            self.assertIn('properties', molecule)
            self.assertIn('toxicity_prediction', molecule)
            self.assertGreater(molecule['toxicity_accuracy'], 0.94)

    def test_drug_repurposing(self):
        """Test drug repurposing and combination therapy prediction"""
        # Create sample drug data
        drug_data = {
            'failed_compounds': [
                {
                    'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O',
                    'original_indication': 'pain',
                    'clinical_data': {
                        'phase': 'II',
                        'failure_reason': 'efficacy',
                        'safety_profile': np.random.rand(10)
                    }
                }
            ],
            'target_pathways': [
                {
                    'pathway_id': 'hsa04010',
                    'proteins': ['P00533', 'P04626'],
                    'interactions': np.random.rand(100, 100)
                }
            ],
            'patient_population': {
                'genetic_markers': np.random.rand(1000, 100),
                'disease_subtypes': np.random.rand(1000, 5)
            }
        }
        
        # Test repurposing analysis
        repurposing_results = self.doctor._analyze_drug_repurposing(drug_data)
        self.assertIsInstance(repurposing_results, dict)
        self.assertIn('viable_candidates', repurposing_results)
        self.assertIn('combination_therapies', repurposing_results)
        self.assertIn('efficacy_predictions', repurposing_results)
        self.assertGreater(repurposing_results['candidate_discovery_rate'], 0.73)

    def test_clinical_trial_optimization(self):
        """Test AGI-driven clinical trial optimization"""
        # Create sample trial data
        trial_data = {
            'biomarkers': {
                'EGFR_mutant': np.random.rand(1000),
                'PD-L1_high': np.random.rand(1000)
            },
            'endpoints': {
                'OS': np.random.rand(1000),
                'PFS': np.random.rand(1000)
            },
            'patient_cohort': {
                'demographics': np.random.rand(1000, 5),
                'medical_history': np.random.rand(1000, 10),
                'genetic_profile': np.random.rand(1000, 100)
            }
        }
        
        # Test trial optimization
        optimized_trial = self.doctor._optimize_clinical_trial(trial_data)
        self.assertIsInstance(optimized_trial, dict)
        self.assertIn('patient_stratification', optimized_trial)
        self.assertIn('dose_regimens', optimized_trial)
        self.assertIn('outcome_predictions', optimized_trial)
        self.assertLess(optimized_trial['predicted_failure_rate'], 0.18)

    def test_federated_learning_integration(self):
        """Test federated learning for drug discovery"""
        # Create sample distributed data
        distributed_data = {
            'biobanks': [
                {
                    'id': 'biobank1',
                    'encrypted_data': np.random.rand(1000, 100),
                    'metadata': {
                        'size': 1000,
                        'diversity_score': 0.85,
                        'quality_metrics': np.random.rand(5)
                    }
                }
            ],
            'global_model': {
                'architecture': 'transformer',
                'parameters': np.random.rand(1000000),
                'performance_metrics': {
                    'accuracy': 0.95,
                    'generalization': 0.92
                }
            }
        }
        
        # Test federated learning
        federated_results = self.doctor._run_federated_learning(distributed_data)
        self.assertIsInstance(federated_results, dict)
        self.assertIn('updated_model', federated_results)
        self.assertIn('aggregated_insights', federated_results)
        self.assertIn('privacy_metrics', federated_results)
        self.assertGreater(federated_results['data_utility'], 0.9)

    def test_bias_mitigation_in_drug_discovery(self):
        """Test bias mitigation in drug discovery process"""
        # Create sample diverse dataset
        diverse_data = {
            'demographic_groups': [
                {
                    'group_id': 'group1',
                    'genetic_data': np.random.rand(1000, 100),
                    'clinical_outcomes': np.random.rand(1000),
                    'representation_score': 0.85
                }
            ],
            'model_predictions': {
                'efficacy': np.random.rand(1000),
                'toxicity': np.random.rand(1000),
                'dosing': np.random.rand(1000)
            }
        }
        
        # Test bias mitigation
        mitigation_results = self.doctor._mitigate_discovery_bias(diverse_data)
        self.assertIsInstance(mitigation_results, dict)
        self.assertIn('fairness_metrics', mitigation_results)
        self.assertIn('adjusted_predictions', mitigation_results)
        self.assertIn('bias_correction_factors', mitigation_results)
        self.assertLess(mitigation_results['outcome_disparity'], 0.02)

if __name__ == '__main__':
    unittest.main() 