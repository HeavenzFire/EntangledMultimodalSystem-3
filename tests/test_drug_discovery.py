import unittest
import numpy as np
from core.protein_structure import QuantumProteinStructurePredictor
from core.molecule_design import QuantumMoleculeGenerator
from core.drug_repurposing import QuantumDrugRepurposer

class TestQuantumDrugDiscovery(unittest.TestCase):
    def setUp(self):
        self.protein_predictor = QuantumProteinStructurePredictor()
        self.molecule_generator = QuantumMoleculeGenerator()
        self.drug_repurposer = QuantumDrugRepurposer()
    
    def test_protein_structure_prediction(self):
        """Test protein structure prediction"""
        sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
        experimental_data = {
            'xray_diffraction': np.random.rand(100, 100),
            'nmr_spectra': np.random.rand(100, 100),
            'cryo_em': np.random.rand(100, 100, 100)
        }
        
        prediction = self.protein_predictor.predict_structure(sequence, experimental_data)
        
        self.assertIsInstance(prediction, dict)
        self.assertIn('3d_structure', prediction)
        self.assertIn('confidence_scores', prediction)
        self.assertIn('binding_sites', prediction)
        self.assertGreater(prediction['accuracy_score'], 0.92)
    
    def test_molecule_generation(self):
        """Test molecule generation"""
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
            },
            'batch_size': 10
        }
        
        molecules = self.molecule_generator.generate_molecules(design_params)
        
        self.assertIsInstance(molecules, list)
        self.assertGreater(len(molecules), 0)
        
        for molecule in molecules:
            self.assertIn('smiles', molecule)
            self.assertIn('properties', molecule)
            self.assertIn('toxicity_prediction', molecule)
            self.assertGreater(molecule['toxicity_accuracy'], 0.94)
    
    def test_drug_repurposing(self):
        """Test drug repurposing analysis"""
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
        
        analysis = self.drug_repurposer.analyze_drug_repurposing(drug_data)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('viable_candidates', analysis)
        self.assertIn('combination_therapies', analysis)
        self.assertIn('efficacy_predictions', analysis)
        self.assertGreater(analysis['candidate_discovery_rate'], 0.73)
    
    def test_combination_therapy_optimization(self):
        """Test combination therapy optimization"""
        candidates = [
            {
                'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O',
                'properties': {
                    'molecular_weight': 180.16,
                    'logp': 1.2,
                    'rotatable_bonds': 2
                }
            }
        ]
        
        optimized_combinations = self.drug_repurposer.optimize_combination_therapy(candidates)
        
        self.assertIsInstance(optimized_combinations, list)
        self.assertGreater(len(optimized_combinations), 0)
        
        for combination in optimized_combinations:
            self.assertIn('smiles', combination)
            self.assertIn('properties', combination)
            self.assertIn('efficacy_score', combination)
    
    def test_molecule_optimization(self):
        """Test molecule optimization"""
        molecules = [
            {
                'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O',
                'properties': {
                    'molecular_weight': 180.16,
                    'logp': 1.2,
                    'rotatable_bonds': 2
                }
            }
        ]
        
        target_profile = {
            'binding_affinity': 0.8,
            'selectivity': 0.9,
            'toxicity_threshold': 0.1
        }
        
        optimized_molecules = self.molecule_generator.optimize_molecules(molecules, target_profile)
        
        self.assertIsInstance(optimized_molecules, list)
        self.assertGreater(len(optimized_molecules), 0)
        
        for molecule in optimized_molecules:
            self.assertIn('smiles', molecule)
            self.assertIn('properties', molecule)
            self.assertIn('optimization_score', molecule)

if __name__ == '__main__':
    unittest.main() 