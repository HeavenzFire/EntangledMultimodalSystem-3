import React, { useState, useEffect } from 'react';
import { Box, Button, Typography, Card, CardContent, List, ListItem, ListItemText } from '@mui/material';

const HomeSafetySwarm = () => {
  const [armed, setArmed] = useState(false);
  const [motionDetected, setMotionDetected] = useState(false);
  const [doorOpen, setDoorOpen] = useState(false);
  const [events, setEvents] = useState([]);

  useEffect(() => {
    if ('Notification' in window) {
      Notification.requestPermission();
    }
  }, []);

  const addEvent = (event) => {
    setEvents(prev => [...prev, `${new Date().toLocaleTimeString()}: ${event}`]);
  };

  const sendAlert = (message) => {
    if (armed && 'Notification' in window && Notification.permission === 'granted') {
      new Notification('Home Safety Alert', { body: message });
    }
    addEvent(message);
  };

  const armSystem = () => {
    setArmed(true);
    addEvent('System armed');
  };

  const disarmSystem = () => {
    setArmed(false);
    addEvent('System disarmed');
  };

  const simulateMotion = () => {
    setMotionDetected(true);
    sendAlert('Motion detected!');
    setTimeout(() => setMotionDetected(false), 5000); // reset after 5s
  };

  const simulateDoorOpen = () => {
    setDoorOpen(true);
    sendAlert('Door opened!');
    setTimeout(() => setDoorOpen(false), 5000); // reset after 5s
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h4" gutterBottom>Home Safety Swarm</Typography>
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6">System Status: {armed ? 'Armed' : 'Disarmed'}</Typography>
          <Typography>Motion Detected: {motionDetected ? 'Yes' : 'No'}</Typography>
          <Typography>Door Open: {doorOpen ? 'Yes' : 'No'}</Typography>
        </CardContent>
      </Card>
      <Box sx={{ mb: 2 }}>
        <Button variant="contained" onClick={armed ? disarmSystem : armSystem} sx={{ mr: 1 }}>
          {armed ? 'Disarm' : 'Arm'} System
        </Button>
        <Button variant="outlined" onClick={simulateMotion} sx={{ mr: 1 }}>Simulate Motion</Button>
        <Button variant="outlined" onClick={simulateDoorOpen}>Simulate Door Open</Button>
      </Box>
      <Typography variant="h6">Event Log</Typography>
      <List>
        {events.map((event, index) => (
          <ListItem key={index}>
            <ListItemText primary={event} />
          </ListItem>
        ))}
      </List>
    </Box>
  );
};

export default HomeSafetySwarm;