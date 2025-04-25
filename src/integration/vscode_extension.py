import json
import websockets
import asyncio
from typing import Dict, List, Any
from pathlib import Path
import logging

class VSCodeIntegration:
    """Handles integration with Visual Studio Code"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.websocket_server = None
        self.connected_clients = set()
        self.logger = logging.getLogger(__name__)
        
    async def start_server(self, host: str = 'localhost', port: int = 8765):
        """Start the WebSocket server for VS Code communication"""
        self.websocket_server = await websockets.serve(
            self._handle_client,
            host,
            port
        )
        self.logger.info(f"VS Code integration server started on {host}:{port}")
        
    async def _handle_client(self, websocket, path):
        """Handle incoming WebSocket connections"""
        self.connected_clients.add(websocket)
        try:
            async for message in websocket:
                await self._process_message(websocket, message)
        finally:
            self.connected_clients.remove(websocket)
            
    async def _process_message(self, websocket, message: str):
        """Process incoming messages from VS Code"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'file_change':
                await self._handle_file_change(data)
            elif message_type == 'collaboration_request':
                await self._handle_collaboration_request(data)
            elif message_type == 'agent_message':
                await self._handle_agent_message(data)
                
        except json.JSONDecodeError:
            self.logger.error("Invalid JSON message received")
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            
    async def _handle_file_change(self, data: Dict[str, Any]):
        """Handle file change notifications from VS Code"""
        file_path = self.workspace_path / data['path']
        if file_path.exists():
            # Notify connected agents about file changes
            await self._broadcast_message({
                'type': 'file_update',
                'path': str(file_path),
                'content': file_path.read_text()
            })
            
    async def _handle_collaboration_request(self, data: Dict[str, Any]):
        """Handle collaboration requests between agents"""
        agent_id = data.get('agent_id')
        if agent_id:
            # Forward collaboration request to specific agent
            await self._send_to_agent(agent_id, {
                'type': 'collaboration_request',
                'request': data.get('request')
            })
            
    async def _handle_agent_message(self, data: Dict[str, Any]):
        """Handle messages between agents"""
        target_agent = data.get('target_agent')
        if target_agent:
            await self._send_to_agent(target_agent, {
                'type': 'agent_message',
                'content': data.get('content')
            })
            
    async def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if self.connected_clients:
            message_str = json.dumps(message)
            await asyncio.gather(
                *[client.send(message_str) for client in self.connected_clients]
            )
            
    async def _send_to_agent(self, agent_id: str, message: Dict[str, Any]):
        """Send message to specific agent"""
        # Implementation depends on how agents are identified
        # This is a placeholder for the actual implementation
        pass
        
    def stop(self):
        """Stop the WebSocket server"""
        if self.websocket_server:
            self.websocket_server.close()
            self.logger.info("VS Code integration server stopped") 