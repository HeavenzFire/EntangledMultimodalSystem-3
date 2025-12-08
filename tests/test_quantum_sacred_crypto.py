import pytest
import numpy as np
from src.quantum.cryptography.einstein_tile import EinsteinTileGenerator, EinsteinTileConfig
from src.quantum.security.sacred_firewall import SacredFirewall, FirewallConfig, ThreatLevel
from src.quantum.security.self_healing import QuantumSelfHealing, HealingConfig, HealingStatus

def test_einstein_tile_generation():
    """Test Einstein Tile pattern generation and key derivation"""
    config = EinsteinTileConfig(
        dimensions=11,
        sacred_ratio=1.618033988749895,
        pattern_size=1024,
        rotation_steps=7
    )
    generator = EinsteinTileGenerator(config)
    
    # Test pattern generation
    pattern = generator.create_pattern()
    assert pattern.shape == (1024, 1024)
    assert np.all(pattern >= 0) and np.all(pattern <= 1)
    
    # Test key generation
    key = generator.generate_key(256)
    assert len(key) == 32  # 256 bits = 32 bytes
    assert isinstance(key, bytes)
    
    # Test pattern verification
    assert generator.verify_pattern(pattern)
    
    # Test pattern modification detection
    modified_pattern = pattern.copy()
    modified_pattern[0, 0] = 0.5
    assert not generator.verify_pattern(modified_pattern)

def test_sacred_firewall():
    """Test quantum-sacred firewall functionality"""
    config = FirewallConfig(
        phi_resonance=1.618033988749895,
        toroidal_cycles=7,
        entropy_threshold=0.8,
        archetype_threshold=0.9
    )
    firewall = SacredFirewall(config)
    
    # Test safe packet validation
    safe_packet = b"Quantum-safe test packet"
    is_valid, threat_level = firewall.validate_packet(safe_packet)
    assert is_valid
    assert threat_level == ThreatLevel.SAFE
    
    # Test malicious packet detection
    malicious_packet = b"Malicious" * 1000
    is_valid, threat_level = firewall.validate_packet(malicious_packet)
    assert not is_valid
    assert threat_level in [ThreatLevel.MALICIOUS, ThreatLevel.CRITICAL]
    
    # Test karmic rebalancing
    for _ in range(10):
        firewall.validate_packet(malicious_packet)
    assert len(firewall.toroidal_shield.entropy_history) == 0

def test_quantum_self_healing():
    """Test quantum-sacred self-healing protocol"""
    config = HealingConfig(
        phi_resonance=1.618033988749895,
        optimization_steps=100,
        convergence_threshold=1e-6,
        sacred_pattern_size=64
    )
    healer = QuantumSelfHealing(config)
    
    # Test healing initialization
    assert healer.status == HealingStatus.INITIALIZED
    
    # Test healing process
    threat_vector = np.random.rand(64)
    result = healer.heal(threat_vector)
    
    assert healer.status == HealingStatus.COMPLETED
    assert 'metrics' in result
    assert 'keys' in result
    assert 'status' in result
    assert result['status'] == 'success'
    
    # Test healing history
    history = healer.get_healing_history()
    assert len(history) > 0
    assert 'threat_vector' in history[0]
    assert 'new_config' in history[0]
    assert 'result' in history[0]
    
    # Test sacred metrics
    metrics = result['metrics']
    assert 'golden_alignment' in metrics
    assert 'pattern_coherence' in metrics
    assert 'entropy' in metrics
    assert 0 <= metrics['golden_alignment'] <= 1
    assert 0 <= metrics['pattern_coherence'] <= 1
    assert 0 <= metrics['entropy'] <= 1
    
    # Test key generation
    keys = result['keys']
    assert 'master_key' in keys
    assert 'encryption_key' in keys
    assert 'signature_key' in keys
    assert len(keys['master_key']) == 64
    assert len(keys['encryption_key']) == 32
    assert len(keys['signature_key']) == 32

def test_integration():
    """Test integration between Einstein Tile, Firewall, and Self-Healing"""
    # Initialize components
    tile_config = EinsteinTileConfig()
    firewall_config = FirewallConfig()
    healing_config = HealingConfig()
    
    generator = EinsteinTileGenerator(tile_config)
    firewall = SacredFirewall(firewall_config)
    healer = QuantumSelfHealing(healing_config)
    
    # Generate quantum key
    key = generator.generate_key()
    
    # Test key validation
    is_valid, threat_level = firewall.validate_packet(key)
    assert is_valid
    assert threat_level == ThreatLevel.SAFE
    
    # Test healing response to threat
    threat_vector = np.random.rand(64)
    healing_result = healer.heal(threat_vector)
    
    # Verify healing metrics
    assert healing_result['metrics']['golden_alignment'] > 0.8
    assert healing_result['metrics']['pattern_coherence'] > 0.8
    assert healing_result['metrics']['entropy'] > 0.7
    
    # Test new key validation
    new_key = healing_result['keys']['master_key']
    is_valid, threat_level = firewall.validate_packet(new_key)
    assert is_valid
    assert threat_level == ThreatLevel.SAFE

def test_edge_cases():
    """Test edge cases and error handling"""
    # Test empty packet
    firewall = SacredFirewall()
    is_valid, threat_level = firewall.validate_packet(b"")
    assert not is_valid
    assert threat_level == ThreatLevel.CRITICAL
    
    # Test invalid threat vector
    healer = QuantumSelfHealing()
    with pytest.raises(RuntimeError):
        healer.heal(np.array([]))
    
    # Test invalid pattern size
    with pytest.raises(ValueError):
        EinsteinTileConfig(pattern_size=0)
    
    # Test invalid sacred ratio
    with pytest.raises(ValueError):
        EinsteinTileConfig(sacred_ratio=0)

def test_performance():
    """Test performance characteristics"""
    import time
    
    # Test Einstein Tile generation performance
    generator = EinsteinTileGenerator()
    start_time = time.time()
    pattern = generator.create_pattern()
    generation_time = time.time() - start_time
    assert generation_time < 5.0  # Should complete within 5 seconds
    
    # Test key generation performance
    start_time = time.time()
    key = generator.generate_key()
    keygen_time = time.time() - start_time
    assert keygen_time < 1.0  # Should complete within 1 second
    
    # Test firewall validation performance
    firewall = SacredFirewall()
    start_time = time.time()
    for _ in range(100):
        firewall.validate_packet(key)
    validation_time = time.time() - start_time
    assert validation_time < 10.0  # Should complete within 10 seconds
    
    # Test healing performance
    healer = QuantumSelfHealing()
    threat_vector = np.random.rand(64)
    start_time = time.time()
    healer.heal(threat_vector)
    healing_time = time.time() - start_time
    assert healing_time < 15.0  # Should complete within 15 seconds 