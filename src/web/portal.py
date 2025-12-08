from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Dict, List, Optional
import json
import asyncio
import logging
from pathlib import Path
from ..agents.intelligent_assistant import IntelligentAssistant

app = FastAPI(title="HyperIntelligent Portal")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Initialize agents
agents: Dict[str, IntelligentAssistant] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    # Create default agent
    default_agent = IntelligentAssistant(
        name="HyperAssistant",
        capabilities=["chat", "learning", "analysis"],
        config={
            "learning_rate": 0.1,
            "personality_traits": {
                "professional": True,
                "friendly": True,
                "detailed": True
            }
        }
    )
    agents["default"] = default_agent

@app.get("/")
async def get_portal():
    """Serve the main portal interface."""
    html_path = static_path / "index.html"
    return HTMLResponse(content=html_path.read_text())

@app.websocket("/ws/chat/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str):
    """Handle WebSocket connections for chat."""
    await websocket.accept()
    
    if agent_id not in agents:
        await websocket.close(code=1008, reason="Agent not found")
        return
    
    agent = agents[agent_id]
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process message with agent
            response = await agent.process_input(message["content"])
            
            # Send response back to client
            await websocket.send_json({
                "type": "response",
                "content": response["response"],
                "timestamp": response["timestamp"]
            })
            
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
        await websocket.close(code=1011, reason=str(e))

@app.post("/api/agents/{agent_id}/preferences")
async def update_agent_preferences(agent_id: str, preferences: Dict):
    """Update agent preferences."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agents[agent_id].update_preferences(preferences)
    return {"status": "success"}

@app.get("/api/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get agent status and statistics."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[agent_id]
    return {
        "status": agent.get_status(),
        "learning_stats": agent.get_learning_stats()
    }

@app.post("/api/agents")
async def create_agent(name: str, capabilities: List[str], config: Optional[Dict] = None):
    """Create a new agent."""
    if name in agents:
        raise HTTPException(status_code=400, detail="Agent already exists")
    
    agent = IntelligentAssistant(name, capabilities, config)
    agents[name] = agent
    return {"status": "success", "agent_id": name}

@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    del agents[agent_id]
    return {"status": "success"} 