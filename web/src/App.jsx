import React, { useState, useEffect } from 'react';
import {
    Box,
    Container,
    Grid,
    Paper,
    Typography,
    TextField,
    Button,
    CircularProgress,
    Alert,
    Chip,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const theme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#90caf9',
        },
        secondary: {
            main: '#f48fb1',
        },
        background: {
            default: '#121212',
            paper: '#1e1e1e',
        },
        error: {
            main: '#f44336',
        },
        warning: {
            main: '#ffa726',
        },
        success: {
            main: '#66bb6a',
        },
    },
});

const API_URL = 'http://localhost:5000/api';

const StatusChip = ({ status }) => {
    const getColor = () => {
        switch (status) {
            case 'valid':
                return 'success';
            case 'warning':
                return 'warning';
            case 'error':
                return 'error';
            case 'critical':
                return 'error';
            default:
                return 'default';
        }
    };

    return (
        <Chip
            label={status.toUpperCase()}
            color={getColor()}
            size="small"
        />
    );
};

function App() {
    const [inputData, setInputData] = useState('');
    const [encryptedData, setEncryptedData] = useState('');
    const [decryptedData, setDecryptedData] = useState('');
    const [metrics, setMetrics] = useState(null);
    const [validationResults, setValidationResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchMetrics = async () => {
        try {
            const response = await fetch(`${API_URL}/quantum/metrics`);
            const data = await response.json();
            setMetrics(data);
        } catch (err) {
            setError(err.message);
        }
    };

    const fetchValidation = async () => {
        try {
            const response = await fetch(`${API_URL}/quantum/validate`);
            const data = await response.json();
            setValidationResults(data);
        } catch (err) {
            setError(err.message);
        }
    };

    useEffect(() => {
        fetchMetrics();
        fetchValidation();
        const interval = setInterval(() => {
            fetchMetrics();
            fetchValidation();
        }, 5000);
        return () => clearInterval(interval);
    }, []);

    const handleEncrypt = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_URL}/quantum/security/encrypt`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ data: inputData }),
            });
            const data = await response.json();
            setEncryptedData(data.encrypted_data);
        } catch (err) {
            setError(err.message);
        }
        setLoading(false);
    };

    const handleDecrypt = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_URL}/quantum/security/decrypt`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    encrypted_data: encryptedData,
                    tag: metrics?.security?.tag,
                    iv: metrics?.security?.iv,
                }),
            });
            const data = await response.json();
            setDecryptedData(data.decrypted_data);
        } catch (err) {
            setError(err.message);
        }
        setLoading(false);
    };

    return (
        <ThemeProvider theme={theme}>
            <Box sx={{ minHeight: '100vh', bgcolor: 'background.default', py: 4 }}>
                <Container maxWidth="lg">
                    <Typography variant="h3" component="h1" gutterBottom align="center" color="primary">
                        Quantum System Interface
                    </Typography>

                    {validationResults && (
                        <Box sx={{ mb: 4 }}>
                            <Typography variant="h6" gutterBottom>
                                System Status
                            </Typography>
                            <TableContainer component={Paper}>
                                <Table>
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Component</TableCell>
                                            <TableCell>Status</TableCell>
                                            <TableCell>Message</TableCell>
                                            <TableCell>Details</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {Object.entries(validationResults).map(([component, result]) => (
                                            <TableRow key={component}>
                                                <TableCell>{component}</TableCell>
                                                <TableCell>
                                                    <StatusChip status={result.status} />
                                                </TableCell>
                                                <TableCell>{result.message}</TableCell>
                                                <TableCell>
                                                    {Object.entries(result.details).map(([key, value]) => (
                                                        <Typography key={key} variant="body2">
                                                            {key}: {value}
                                                        </Typography>
                                                    ))}
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </Box>
                    )}

                    <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Data Encryption
                                </Typography>
                                <TextField
                                    fullWidth
                                    multiline
                                    rows={4}
                                    value={inputData}
                                    onChange={(e) => setInputData(e.target.value)}
                                    placeholder="Enter data to encrypt"
                                    margin="normal"
                                />
                                <Button
                                    variant="contained"
                                    onClick={handleEncrypt}
                                    disabled={loading || !inputData}
                                    sx={{ mt: 2 }}
                                >
                                    Encrypt
                                </Button>
                            </Paper>
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Data Decryption
                                </Typography>
                                <TextField
                                    fullWidth
                                    multiline
                                    rows={4}
                                    value={encryptedData}
                                    onChange={(e) => setEncryptedData(e.target.value)}
                                    placeholder="Encrypted data"
                                    margin="normal"
                                />
                                <Button
                                    variant="contained"
                                    onClick={handleDecrypt}
                                    disabled={loading || !encryptedData}
                                    sx={{ mt: 2 }}
                                >
                                    Decrypt
                                </Button>
                                {decryptedData && (
                                    <Typography sx={{ mt: 2 }} color="text.secondary">
                                        Decrypted: {decryptedData}
                                    </Typography>
                                )}
                            </Paper>
                        </Grid>

                        <Grid item xs={12}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    System Metrics
                                </Typography>
                                {metrics ? (
                                    <LineChart
                                        width={800}
                                        height={300}
                                        data={[
                                            {
                                                name: 'Security',
                                                value: metrics.security?.key_strength || 0,
                                            },
                                            {
                                                name: 'Synthesis',
                                                value: metrics.synthesis?.coherence || 0,
                                            },
                                            {
                                                name: 'Torus',
                                                value: metrics.torus?.harmony || 0,
                                            },
                                        ]}
                                    >
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="name" />
                                        <YAxis />
                                        <Tooltip />
                                        <Legend />
                                        <Line type="monotone" dataKey="value" stroke="#8884d8" />
                                    </LineChart>
                                ) : (
                                    <CircularProgress />
                                )}
                            </Paper>
                        </Grid>
                    </Grid>

                    {error && (
                        <Alert severity="error" sx={{ mt: 3 }}>
                            {error}
                        </Alert>
                    )}
                </Container>
            </Box>
        </ThemeProvider>
    );
}

export default App; 