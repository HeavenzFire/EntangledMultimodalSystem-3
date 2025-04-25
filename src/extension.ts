import * as vscode from 'vscode';
import * as WebSocket from 'ws';
import { QuantumVisualization } from './extension/quantum_visualization';

interface AgentMessage {
    type: string;
    content?: any;
    target_agent?: string;
    agent_id?: string;
    quantum_state?: any;
}

export class HyperIntelligentCollaboration {
    private ws: WebSocket | null = null;
    private context: vscode.ExtensionContext;
    private statusBarItem: vscode.StatusBarItem;
    private isConnected: boolean = false;
    private quantumVisualization: QuantumVisualization;
    private collaborationPanel: vscode.WebviewPanel | undefined;
    
    constructor(context: vscode.ExtensionContext) {
        this.context = context;
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Left,
            100
        );
        this.statusBarItem.text = "$(sync) HyperIntelligent";
        this.statusBarItem.show();
        
        this.quantumVisualization = new QuantumVisualization(context);
        
        this.registerCommands();
        this.setupEventListeners();
    }
    
    private registerCommands() {
        let startCommand = vscode.commands.registerCommand(
            'hyperintelligent.startCollaboration',
            () => this.startCollaboration()
        );
        
        let stopCommand = vscode.commands.registerCommand(
            'hyperintelligent.stopCollaboration',
            () => this.stopCollaboration()
        );
        
        let inviteCommand = vscode.commands.registerCommand(
            'hyperintelligent.inviteAgent',
            () => this.inviteAgent()
        );
        
        let showVisualizationCommand = vscode.commands.registerCommand(
            'hyperintelligent.showVisualization',
            () => this.showQuantumVisualization()
        );
        
        this.context.subscriptions.push(
            startCommand,
            stopCommand,
            inviteCommand,
            showVisualizationCommand
        );
    }
    
    private setupEventListeners() {
        // Listen for file changes
        vscode.workspace.onDidChangeTextDocument((event) => {
            if (this.isConnected && this.ws) {
                this.sendMessage({
                    type: 'file_change',
                    path: event.document.uri.fsPath,
                    content: event.document.getText()
                });
            }
        });
        
        // Listen for configuration changes
        vscode.workspace.onDidChangeConfiguration((event) => {
            if (event.affectsConfiguration('hyperintelligent')) {
                this.updateConfiguration();
            }
        });
    }
    
    private async startCollaboration() {
        const config = vscode.workspace.getConfiguration('hyperintelligent');
        const host = config.get('serverHost', 'localhost');
        const port = config.get('serverPort', 8765);
        const agentId = config.get('agentId');
        
        if (!agentId) {
            vscode.window.showErrorMessage('Please set your agent ID in settings');
            return;
        }
        
        try {
            this.ws = new WebSocket(`ws://${host}:${port}`);
            
            this.ws.on('open', () => {
                this.isConnected = true;
                this.statusBarItem.text = "$(sync~spin) HyperIntelligent";
                vscode.window.showInformationMessage('Connected to collaboration server');
                
                this.sendMessage({
                    type: 'agent_connect',
                    agent_id: agentId
                });
            });
            
            this.ws.on('message', (data: WebSocket.Data) => {
                this.handleMessage(data.toString());
            });
            
            this.ws.on('close', () => {
                this.isConnected = false;
                this.statusBarItem.text = "$(sync) HyperIntelligent";
                vscode.window.showInformationMessage('Disconnected from collaboration server');
            });
            
            this.ws.on('error', (error) => {
                vscode.window.showErrorMessage(`WebSocket error: ${error.message}`);
            });
            
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to connect: ${error}`);
        }
    }
    
    private stopCollaboration() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
    
    private async inviteAgent() {
        const agentId = await vscode.window.showInputBox({
            prompt: 'Enter the agent ID to invite',
            placeHolder: 'agent-id'
        });
        
        if (agentId) {
            this.sendMessage({
                type: 'collaboration_request',
                target_agent: agentId
            });
        }
    }
    
    private showQuantumVisualization() {
        this.quantumVisualization.updateQuantumState(this.getCurrentQuantumState());
    }
    
    private getCurrentQuantumState() {
        // Get current quantum state from the server
        // This is a placeholder - implement actual quantum state retrieval
        return {
            amplitude: 1.0,
            phase: 0.0,
            energy: 1.0
        };
    }
    
    private sendMessage(message: AgentMessage) {
        if (this.ws && this.isConnected) {
            this.ws.send(JSON.stringify(message));
        }
    }
    
    private handleMessage(data: string) {
        try {
            const message: AgentMessage = JSON.parse(data);
            
            switch (message.type) {
                case 'file_update':
                    this.handleFileUpdate(message);
                    break;
                case 'collaboration_request':
                    this.handleCollaborationRequest(message);
                    break;
                case 'agent_message':
                    this.handleAgentMessage(message);
                    break;
                case 'quantum_state_update':
                    this.handleQuantumStateUpdate(message);
                    break;
            }
        } catch (error) {
            console.error('Error handling message:', error);
        }
    }
    
    private async handleFileUpdate(message: AgentMessage) {
        if (message.content && message.path) {
            const uri = vscode.Uri.file(message.path);
            const edit = new vscode.WorkspaceEdit();
            edit.replace(uri, new vscode.Range(0, 0, 0, 0), message.content);
            await vscode.workspace.applyEdit(edit);
        }
    }
    
    private async handleCollaborationRequest(message: AgentMessage) {
        const response = await vscode.window.showQuickPick(
            ['Accept', 'Decline'],
            { placeHolder: `Accept collaboration request from ${message.agent_id}?` }
        );
        
        if (response === 'Accept') {
            this.sendMessage({
                type: 'collaboration_response',
                target_agent: message.agent_id,
                accepted: true
            });
            
            // Show collaboration panel
            this.showCollaborationPanel(message.agent_id);
        }
    }
    
    private handleAgentMessage(message: AgentMessage) {
        if (message.content) {
            vscode.window.showInformationMessage(
                `Message from ${message.agent_id}: ${message.content}`
            );
        }
    }
    
    private handleQuantumStateUpdate(message: AgentMessage) {
        if (message.quantum_state) {
            this.quantumVisualization.updateQuantumState(message.quantum_state);
        }
    }
    
    private showCollaborationPanel(agentId: string) {
        if (this.collaborationPanel) {
            this.collaborationPanel.reveal();
            return;
        }
        
        this.collaborationPanel = vscode.window.createWebviewPanel(
            'collaboration',
            `Collaboration with ${agentId}`,
            vscode.ViewColumn.Two,
            {
                enableScripts: true
            }
        );
        
        this.collaborationPanel.webview.html = this.getCollaborationPanelHtml(agentId);
        
        this.collaborationPanel.onDidDispose(() => {
            this.collaborationPanel = undefined;
        });
    }
    
    private getCollaborationPanelHtml(agentId: string): string {
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Collaboration with ${agentId}</title>
                <style>
                    body {
                        padding: 20px;
                        font-family: Arial, sans-serif;
                    }
                    .chat-container {
                        height: 300px;
                        overflow-y: auto;
                        border: 1px solid #ccc;
                        padding: 10px;
                        margin-bottom: 10px;
                    }
                    .message {
                        margin-bottom: 10px;
                        padding: 5px;
                        border-radius: 5px;
                    }
                    .message.sent {
                        background-color: #e3f2fd;
                        text-align: right;
                    }
                    .message.received {
                        background-color: #f5f5f5;
                    }
                </style>
            </head>
            <body>
                <h2>Collaboration with ${agentId}</h2>
                <div class="chat-container" id="chatContainer"></div>
                <input type="text" id="messageInput" placeholder="Type a message...">
                <button onclick="sendMessage()">Send</button>
                
                <script>
                    const vscode = acquireVsCodeApi();
                    const chatContainer = document.getElementById('chatContainer');
                    const messageInput = document.getElementById('messageInput');
                    
                    function sendMessage() {
                        const message = messageInput.value;
                        if (message) {
                            vscode.postMessage({
                                type: 'agent_message',
                                target_agent: '${agentId}',
                                content: message
                            });
                            
                            addMessage(message, true);
                            messageInput.value = '';
                        }
                    }
                    
                    function addMessage(message, isSent) {
                        const messageDiv = document.createElement('div');
                        messageDiv.className = \`message \${isSent ? 'sent' : 'received'}\`;
                        messageDiv.textContent = message;
                        chatContainer.appendChild(messageDiv);
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                    
                    window.addEventListener('message', event => {
                        const message = event.data;
                        if (message.type === 'agent_message') {
                            addMessage(message.content, false);
                        }
                    });
                </script>
            </body>
            </html>
        `;
    }
    
    private updateConfiguration() {
        // Handle configuration updates
        const config = vscode.workspace.getConfiguration('hyperintelligent');
        // Update any necessary settings
    }
    
    dispose() {
        this.stopCollaboration();
        this.statusBarItem.dispose();
        this.quantumVisualization.dispose();
        if (this.collaborationPanel) {
            this.collaborationPanel.dispose();
        }
    }
}

export function activate(context: vscode.ExtensionContext) {
    new HyperIntelligentCollaboration(context);
}

export function deactivate() {
    // Cleanup if needed
} 