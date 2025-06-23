import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.core.ethical_dao import EthicalDAO, QuantumResistantLedger, AsilomarPrinciplesV5

@pytest.fixture
def dao():
    """Create an EthicalDAO instance for testing."""
    return EthicalDAO()

def test_initialization(dao):
    """Test DAO initialization."""
    assert isinstance(dao.blockchain, QuantumResistantLedger)
    assert isinstance(dao.constitution, AsilomarPrinciplesV5)
    assert len(dao.members) == 0
    assert len(dao.proposals) == 0
    assert len(dao.blockchain.blocks) == 1  # Genesis block

def test_add_member(dao):
    """Test member addition."""
    member_address = "0x1234567890abcdef"
    dao.add_member(member_address)
    
    assert member_address in dao.members
    assert len(dao.blockchain.blocks) == 2  # Genesis + member addition

def test_create_proposal(dao):
    """Test proposal creation."""
    proposal_data = {
        "title": "Test Proposal",
        "description": "This is a test proposal",
        "type": "governance"
    }
    
    proposal_hash = dao.create_proposal(proposal_data)
    
    assert proposal_hash in dao.proposals
    assert dao.proposals[proposal_hash]["data"] == proposal_data
    assert dao.proposals[proposal_hash]["status"] == "pending"
    assert len(dao.blockchain.blocks) == 2  # Genesis + proposal creation

def test_vote(dao):
    """Test voting on a proposal."""
    # First create a proposal
    proposal_data = {"title": "Vote Test"}
    proposal_hash = dao.create_proposal(proposal_data)
    
    # Cast votes
    result = dao.vote(proposal_hash)
    
    assert "compliance_scores" in result
    assert "overall_compliance" in result
    assert "principles_violated" in result
    assert len(dao.proposals[proposal_hash]["votes"]) > 0
    assert len(dao.blockchain.blocks) == 3  # Genesis + proposal + vote

def test_get_proposal_status(dao):
    """Test proposal status retrieval."""
    proposal_data = {"title": "Status Test"}
    proposal_hash = dao.create_proposal(proposal_data)
    
    status = dao.get_proposal_status(proposal_hash)
    
    assert status["data"] == proposal_data
    assert status["status"] == "pending"
    assert "created" in status

def test_get_state(dao):
    """Test state retrieval."""
    # Add some test data
    dao.add_member("0x1")
    dao.add_member("0x2")
    dao.create_proposal({"title": "Test 1"})
    dao.create_proposal({"title": "Test 2"})
    
    state = dao.get_state()
    
    assert state["member_count"] == 2
    assert state["proposal_count"] == 2
    assert state["block_height"] == 4  # Genesis + 2 members + 2 proposals
    assert state["active_proposals"] == 2

def test_reset(dao):
    """Test DAO reset."""
    # Add some test data
    dao.add_member("0x1")
    dao.create_proposal({"title": "Test"})
    
    # Reset the DAO
    dao.reset()
    
    assert len(dao.members) == 0
    assert len(dao.proposals) == 0
    assert len(dao.blockchain.blocks) == 1  # Only genesis block remains

def test_quantum_resistant_ledger():
    """Test QuantumResistantLedger functionality."""
    ledger = QuantumResistantLedger()
    
    assert len(ledger.blocks) == 1  # Genesis block
    assert ledger.blocks[0]["index"] == 0
    assert ledger.blocks[0]["previous_hash"] == "0" * 64
    
    # Test block addition
    test_data = {"test": "data"}
    ledger.add_block(test_data)
    
    assert len(ledger.blocks) == 2
    assert ledger.blocks[1]["data"] == test_data
    assert ledger.blocks[1]["previous_hash"] == ledger.blocks[0]["hash"]

def test_asilomar_principles():
    """Test AsilomarPrinciplesV5 functionality."""
    principles = AsilomarPrinciplesV5()
    
    # Test principle validation
    votes = [0.8, 0.9, 0.7, 0.6]  # Sample votes
    result = principles.validate(votes)
    
    assert "compliance_scores" in result
    assert "overall_compliance" in result
    assert "principles_violated" in result
    assert isinstance(result["overall_compliance"], float)
    assert 0 <= result["overall_compliance"] <= 1

def test_error_handling(dao):
    """Test error handling in DAO operations."""
    # Test invalid proposal hash
    with pytest.raises(ValueError):
        dao.get_proposal_status("invalid_hash")
    
    with pytest.raises(ValueError):
        dao.vote("invalid_hash")
    
    # Test duplicate member addition
    member = "0x123"
    dao.add_member(member)
    dao.add_member(member)  # Should not raise error, just ignore
    
    assert len(dao.members) == 1  # Still only one unique member

def test_principle_weights():
    """Test principle weighting system."""
    principles = AsilomarPrinciplesV5()
    
    # Test weight retrieval
    assert principles._get_principle_weight("safety") == 1.0
    assert principles._get_principle_weight("value_alignment") == 1.0
    assert principles._get_principle_weight("research_goal") == 0.8  # Default weight
    
    # Test weighted voting
    votes = [1.0] * 100  # All votes are 1.0
    result = principles.validate(votes)
    
    # High-weight principles should have higher scores
    assert result["compliance_scores"]["safety"] >= result["compliance_scores"]["research_goal"] 