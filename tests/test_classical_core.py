import pytest
import numpy as np
import torch
from core.classical.fractal_engine import (
    LSystemGrammar,
    TensorNetworkRenderer,
    FractalEngine
)

def test_lsystem_grammar():
    """Test LSystemGrammar implementation."""
    grammar = LSystemGrammar()
    
    # Test default rules
    assert grammar.alphabet == "F+-[]"
    assert len(grammar.custom_symbols) == 8
    assert all(symbol in grammar.production_rules for symbol in grammar.alphabet)
    
    # Test rule application
    result = grammar.apply_rules("F", iterations=2)
    assert result == "FFFF"  # F -> FF -> FFFF
    
    # Test complexity calculation
    complexity = grammar.calculate_kolmogorov_complexity("F+F-F")
    assert isinstance(complexity, float)
    assert complexity > 0

def test_tensor_network_renderer():
    """Test TensorNetworkRenderer implementation."""
    renderer = TensorNetworkRenderer(bond_dim=64, truncation_error=1e-5)
    
    # Test initialization
    assert renderer.bond_dim == 64
    assert len(renderer.tensors) == 10
    
    # Test forward pass
    x = torch.randn(100, 64)
    output = renderer(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 64
    
    # Test fractal rendering
    lsystem_string = "F[+F]F[-F]F"
    image = renderer.render_fractal(lsystem_string)
    assert isinstance(image, np.ndarray)

def test_fractal_engine():
    """Test FractalEngine implementation."""
    engine = FractalEngine()
    
    # Test fractal generation
    lsystem_string, image = engine.generate_fractal("F", iterations=3)
    assert isinstance(lsystem_string, str)
    assert isinstance(image, np.ndarray)
    
    # Test grammar optimization
    target_complexity = 2.0
    optimized_rules = engine.optimize_grammar(target_complexity)
    assert isinstance(optimized_rules, dict)
    assert all(isinstance(k, str) and isinstance(v, str) 
              for k, v in optimized_rules.items())

def test_grammar_optimization():
    """Test grammar optimization with specific targets."""
    engine = FractalEngine()
    
    # Test optimization with different target complexities
    for target in [1.5, 2.0, 2.5]:
        rules = engine.optimize_grammar(target)
        complexity = engine.grammar.calculate_kolmogorov_complexity(
            engine.grammar.apply_rules("F", 3)
        )
        # Allow for some tolerance in the optimization
        assert abs(complexity - target) < 0.5

def test_tensor_network_visualization():
    """Test tensor network visualization capabilities."""
    renderer = TensorNetworkRenderer()
    
    # Test with different L-system strings
    test_strings = [
        "F",
        "F[+F]F[-F]F",
        "F[+F][-F]F[+F][-F]F"
    ]
    
    for string in test_strings:
        image = renderer.render_fractal(string)
        assert image.shape[0] > 0
        assert image.shape[1] > 0
        assert np.any(image != 0)  # Ensure image is not all zeros

def test_grammar_rule_application():
    """Test grammar rule application with custom rules."""
    grammar = LSystemGrammar()
    
    # Test with custom rules
    custom_rules = {
        'F': 'F[+F]F[-F]F',
        '+': '++',
        '-': '--',
        '[': '[',
        ']': ']'
    }
    grammar.production_rules = custom_rules
    
    # Test multiple iterations
    result = grammar.apply_rules("F", iterations=2)
    assert len(result) > len("F")  # Should be longer than initial string
    assert '[' in result and ']' in result  # Should contain brackets
    
    # Test complexity with custom rules
    complexity = grammar.calculate_kolmogorov_complexity(result)
    assert complexity > 0 