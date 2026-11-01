from types import SimpleNamespace

from src.core.system_optimizer import SystemOptimizer


def test_optimizer_reuses_injected_manager_components():
    component_names = (
        "engine",
        "orchestrator",
        "nexus",
        "consciousness",
        "ethical_governor",
        "multimodal_gan",
        "quantum_interface",
        "holographic_interface",
        "neural_interface",
        "controller",
        "coordinator",
        "integrator",
        "architect",
        "analyzer",
        "evaluator",
        "director",
        "planner",
        "scheduler",
        "executor",
        "monitor",
        "validator",
    )
    components = {name: object() for name in component_names}
    manager = SimpleNamespace(**components)

    optimizer = SystemOptimizer(system_manager=manager)

    assert optimizer.manager is manager
    for name, component in components.items():
        assert getattr(optimizer, name) is component
