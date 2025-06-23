/**
 * Entangled Multimodal System VS Code Extension
 * 
 * This extension provides integration between VS Code and the Entangled Multimodal System,
 * enabling enhanced coding capabilities through quantum computing and consciousness field manipulation.
 */

const vscode = require('vscode');
const net = require('net');
const path = require('path');
const fs = require('fs');

// Configuration
let config = {
    port: 9735,
    autoConnect: true,
    quantumCompletionEnabled: true,
    fractalRefactoringEnabled: true
};

// Connection state
let socket = null;
let connected = false;
let multiagentSystem = null;

/**
 * Activate the extension
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
    console.log('Entangled Multimodal System extension is now active');

    // Load configuration
    loadConfiguration();

    // Register commands
    let connectCommand = vscode.commands.registerCommand('entangled-multimodal.connect', connectToSystem);
    let quantumCompletionCommand = vscode.commands.registerCommand('entangled-multimodal.quantumCompletion', quantumCompletion);
    let fractalRefactoringCommand = vscode.commands.registerCommand('entangled-multimodal.fractalRefactoring', fractalRefactoring);
    let delegateToAgentCommand = vscode.commands.registerCommand('entangled-multimodal.delegateToAgent', delegateToAgent);
    let visualizeQuantumStateCommand = vscode.commands.registerCommand('entangled-multimodal.visualizeQuantumState', visualizeQuantumState);
    let activatePyramidReactivationCommand = vscode.commands.registerCommand('entangled-multimodal.activatePyramidReactivation', activatePyramidReactivation);
    let runQuantumSovereigntyCommand = vscode.commands.registerCommand('entangled-multimodal.runQuantumSovereignty', runQuantumSovereignty);
    let initializeTawhidCircuitCommand = vscode.commands.registerCommand('entangled-multimodal.initializeTawhidCircuit', initializeTawhidCircuit);
    let initializeProphetQubitArrayCommand = vscode.commands.registerCommand('entangled-multimodal.initializeProphetQubitArray', initializeProphetQubitArray);
    let runQuantumVisualizationCommand = vscode.commands.registerCommand('entangled-multimodal.runQuantumVisualization', runQuantumVisualization);

    // Add commands to subscriptions
    context.subscriptions.push(connectCommand);
    context.subscriptions.push(quantumCompletionCommand);
    context.subscriptions.push(fractalRefactoringCommand);
    context.subscriptions.push(delegateToAgentCommand);
    context.subscriptions.push(visualizeQuantumStateCommand);
    context.subscriptions.push(activatePyramidReactivationCommand);
    context.subscriptions.push(runQuantumSovereigntyCommand);
    context.subscriptions.push(initializeTawhidCircuitCommand);
    context.subscriptions.push(initializeProphetQubitArrayCommand);
    context.subscriptions.push(runQuantumVisualizationCommand);

    // Auto-connect if enabled
    if (config.autoConnect) {
        connectToSystem();
    }

    // Register status bar item
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.text = "$(radio-tower) Entangled System";
    statusBarItem.command = 'entangled-multimodal.connect';
    statusBarItem.tooltip = 'Connect to Entangled Multimodal System';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Update status bar based on connection state
    function updateStatusBar() {
        if (connected) {
            statusBarItem.text = "$(radio-tower) Entangled System: Connected";
            statusBarItem.tooltip = 'Connected to Entangled Multimodal System';
        } else {
            statusBarItem.text = "$(radio-tower) Entangled System: Disconnected";
            statusBarItem.tooltip = 'Connect to Entangled Multimodal System';
        }
    }

    // Register configuration change listener
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('entangled-multimodal')) {
                loadConfiguration();
            }
        })
    );

    // Load configuration from VS Code settings
    function loadConfiguration() {
        const settings = vscode.workspace.getConfiguration('entangled-multimodal');
        config.port = settings.get('port', 9735);
        config.autoConnect = settings.get('autoConnect', true);
        config.quantumCompletionEnabled = settings.get('quantumCompletionEnabled', true);
        config.fractalRefactoringEnabled = settings.get('fractalRefactoringEnabled', true);
    }

    // Connect to the Entangled Multimodal System
    function connectToSystem() {
        if (connected) {
            vscode.window.showInformationMessage('Already connected to Entangled Multimodal System');
            return;
        }

        vscode.window.showInformationMessage('Connecting to Entangled Multimodal System...');

        // Create socket connection
        socket = new net.Socket();
        
        socket.connect(config.port, 'localhost', () => {
            connected = true;
            updateStatusBar();
            vscode.window.showInformationMessage('Connected to Entangled Multimodal System');
            
            // Send connection message
            sendMessage({
                type: 'connection',
                client: 'vscode-extension',
                version: '1.0.0'
            });
        });

        socket.on('data', (data) => {
            try {
                const message = JSON.parse(data.toString());
                handleMessage(message);
            } catch (error) {
                console.error('Error parsing message:', error);
            }
        });

        socket.on('close', () => {
            connected = false;
            updateStatusBar();
            vscode.window.showInformationMessage('Disconnected from Entangled Multimodal System');
        });

        socket.on('error', (error) => {
            connected = false;
            updateStatusBar();
            vscode.window.showErrorMessage(`Connection error: ${error.message}`);
        });
    }

    // Send message to the Entangled Multimodal System
    function sendMessage(message) {
        if (!connected || !socket) {
            vscode.window.showErrorMessage('Not connected to Entangled Multimodal System');
            return;
        }

        try {
            socket.write(JSON.stringify(message) + '\n');
        } catch (error) {
            vscode.window.showErrorMessage(`Error sending message: ${error.message}`);
        }
    }

    // Handle messages from the Entangled Multimodal System
    function handleMessage(message) {
        const type = message.type;
        
        switch (type) {
            case 'completion':
                handleCompletion(message);
                break;
            case 'refactoring':
                handleRefactoring(message);
                break;
            case 'visualization':
                handleVisualization(message);
                break;
            case 'notification':
                handleNotification(message);
                break;
            default:
                console.log('Unknown message type:', type);
        }
    }

    // Handle completion response
    function handleCompletion(message) {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const position = message.position;
        const completions = message.completions;

        // Show completions in VS Code
        vscode.window.showQuickPick(completions, {
            placeHolder: 'Select completion'
        }).then(selected => {
            if (selected) {
                editor.edit(editBuilder => {
                    const pos = new vscode.Position(position.line, position.character);
                    editBuilder.insert(pos, selected);
                });
            }
        });
    }

    // Handle refactoring response
    function handleRefactoring(message) {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const refactoredCode = message.refactored;
        
        // Show diff editor with refactored code
        const originalUri = editor.document.uri;
        const refactoredUri = originalUri.with({ scheme: 'refactored' });
        
        vscode.workspace.openTextDocument(refactoredUri).then(doc => {
            vscode.commands.executeCommand('vscode.diff', originalUri, refactoredUri, 'Original ↔ Refactored');
        });
    }

    // Handle visualization response
    function handleVisualization(message) {
        const visualizationData = message.data;
        
        // Create webview panel to display visualization
        const panel = vscode.window.createWebviewPanel(
            'quantumVisualization',
            'Quantum Visualization',
            vscode.ViewColumn.Two,
            {
                enableScripts: true
            }
        );
        
        // Set HTML content for visualization
        panel.webview.html = getWebviewContent(visualizationData);
    }

    // Handle notification
    function handleNotification(message) {
        const notification = message.notification;
        
        switch (notification.type) {
            case 'info':
                vscode.window.showInformationMessage(notification.message);
                break;
            case 'warning':
                vscode.window.showWarningMessage(notification.message);
                break;
            case 'error':
                vscode.window.showErrorMessage(notification.message);
                break;
        }
    }

    // Get webview content for visualization
    function getWebviewContent(data) {
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Quantum Visualization</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                }
                .visualization {
                    width: 100%;
                    height: 500px;
                    border: 1px solid #333;
                    background-color: #252526;
                }
                .controls {
                    margin-top: 20px;
                }
                button {
                    background-color: #0e639c;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    margin-right: 10px;
                    cursor: pointer;
                }
                button:hover {
                    background-color: #1177bb;
                }
            </style>
        </head>
        <body>
            <h1>Quantum Visualization</h1>
            <div class="visualization" id="visualization"></div>
            <div class="controls">
                <button id="rotate">Rotate</button>
                <button id="zoom-in">Zoom In</button>
                <button id="zoom-out">Zoom Out</button>
                <button id="reset">Reset</button>
            </div>
            <script>
                // Visualization script would go here
                // This is a placeholder for the actual visualization code
                document.getElementById('visualization').innerHTML = 'Quantum visualization would be rendered here';
            </script>
        </body>
        </html>`;
    }

    // Quantum code completion
    function quantumCompletion() {
        if (!config.quantumCompletionEnabled) {
            vscode.window.showWarningMessage('Quantum completion is disabled');
            return;
        }

        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        const position = editor.selection.active;
        const document = editor.document;
        const text = document.getText();
        
        sendMessage({
            type: 'request',
            id: generateId(),
            method: 'quantumCompletion',
            params: {
                code: text,
                position: {
                    line: position.line,
                    character: position.character
                }
            }
        });
    }

    // Fractal code refactoring
    function fractalRefactoring() {
        if (!config.fractalRefactoringEnabled) {
            vscode.window.showWarningMessage('Fractal refactoring is disabled');
            return;
        }

        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        const document = editor.document;
        const text = document.getText();
        
        sendMessage({
            type: 'request',
            id: generateId(),
            method: 'fractalRefactoring',
            params: {
                code: text,
                options: {
                    complexity: 0.8,
                    fractal_depth: 3
                }
            }
        });
    }

    // Delegate task to agent
    function delegateToAgent() {
        vscode.window.showInputBox({
            prompt: 'Enter agent name',
            placeHolder: 'e.g., quantum-processor, consciousness-field'
        }).then(agentName => {
            if (!agentName) return;
            
            vscode.window.showInputBox({
                prompt: 'Enter task description',
                placeHolder: 'e.g., Analyze this code for quantum patterns'
            }).then(taskDescription => {
                if (!taskDescription) return;
                
                sendMessage({
                    type: 'request',
                    id: generateId(),
                    method: 'delegateToAgent',
                    params: {
                        agent: agentName,
                        task: {
                            description: taskDescription
                        }
                    }
                });
            });
        });
    }

    // Visualize quantum state
    function visualizeQuantumState() {
        sendMessage({
            type: 'request',
            id: generateId(),
            method: 'visualizeQuantumState',
            params: {}
        });
    }

    // Activate pyramid reactivation
    function activatePyramidReactivation() {
        sendMessage({
            type: 'request',
            id: generateId(),
            method: 'activatePyramidReactivation',
            params: {}
        });
    }

    // Run quantum sovereignty protocol
    function runQuantumSovereignty() {
        sendMessage({
            type: 'request',
            id: generateId(),
            method: 'runQuantumSovereignty',
            params: {}
        });
    }

    // Initialize Tawhid Circuit
    function initializeTawhidCircuit() {
        sendMessage({
            type: 'request',
            id: generateId(),
            method: 'initializeTawhidCircuit',
            params: {}
        });
    }

    // Initialize Prophet Qubit Array
    function initializeProphetQubitArray() {
        vscode.window.showInputBox({
            prompt: 'Enter Tawhid Circuit ID',
            placeHolder: 'e.g., tawhid-001'
        }).then(circuitId => {
            if (!circuitId) return;
            
            sendMessage({
                type: 'request',
                id: generateId(),
                method: 'initializeProphetQubitArray',
                params: {
                    tawhid_circuit_id: circuitId
                }
            });
        });
    }

    // Run quantum visualization
    function runQuantumVisualization() {
        vscode.window.showInputBox({
            prompt: 'Enter Tawhid Circuit ID',
            placeHolder: 'e.g., tawhid-001'
        }).then(circuitId => {
            if (!circuitId) return;
            
            vscode.window.showInputBox({
                prompt: 'Enter Prophet Qubit Array ID',
                placeHolder: 'e.g., prophet-001'
            }).then(arrayId => {
                if (!arrayId) return;
                
                sendMessage({
                    type: 'request',
                    id: generateId(),
                    method: 'runQuantumVisualization',
                    params: {
                        tawhid_circuit_id: circuitId,
                        prophet_array_id: arrayId
                    }
                });
            });
        });
    }

    // Generate unique ID for requests
    function generateId() {
        return 'req-' + Math.random().toString(36).substr(2, 9);
    }
}

/**
 * Deactivate the extension
 */
function deactivate() {
    if (socket) {
        socket.destroy();
    }
}

module.exports = {
    activate,
    deactivate
}; 