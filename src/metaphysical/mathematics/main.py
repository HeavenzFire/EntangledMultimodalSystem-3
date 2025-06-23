import logging
import argparse
from typing import Optional
import sys
import signal
import time

from .core.safeguard_orchestrator import SafeguardOrchestrator
from .core.monitoring import SystemMonitor
from .dashboard.app import app
from .config.safeguard_config import DEFAULT_CONFIG, update_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('safeguard_system.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Divine Digital Twin Safeguard System')
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--state-dim', type=int, default=64, help='State dimension')
    parser.add_argument('--update-interval', type=float, default=1.0, help='Update interval in seconds')
    parser.add_argument('--dashboard', action='store_true', help='Start dashboard')
    parser.add_argument('--monitor', action='store_true', help='Start monitoring')
    
    return parser.parse_args()

def initialize_system(args: argparse.Namespace) -> tuple[SafeguardOrchestrator, Optional[SystemMonitor]]:
    """Initialize the safeguard system"""
    try:
        # Update configuration if provided
        if args.config:
            # Load configuration from file
            # This would be implemented based on the configuration file format
            pass
            
        # Update default configuration
        update_config('orchestrator', state_dim=args.state_dim)
        update_config('orchestrator', update_interval=args.update_interval)
        
        # Initialize orchestrator
        orchestrator = SafeguardOrchestrator(state_dim=args.state_dim)
        
        # Initialize monitor if requested
        monitor = None
        if args.monitor:
            monitor = SystemMonitor(orchestrator)
            monitor.start_monitoring()
            
        return orchestrator, monitor
        
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        raise

def run_dashboard(app) -> None:
    """Run the dashboard application"""
    try:
        app.run_server(debug=True, port=8050)
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        raise

def handle_shutdown(orchestrator: SafeguardOrchestrator, monitor: Optional[SystemMonitor]) -> None:
    """Handle system shutdown"""
    try:
        logger.info("Shutting down system...")
        
        # Stop monitoring if active
        if monitor:
            monitor.stop_monitoring()
            
        # Get final system report
        report = orchestrator.get_orchestration_report()
        logger.info(f"Final System Status: {report['system_status']}")
        logger.info(f"Overall Safeguard Score: {report['overall_safeguard_score']:.2f}")
        
        logger.info("System shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
        raise

def main() -> None:
    """Main entry point"""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Initialize system
        orchestrator, monitor = initialize_system(args)
        
        # Set up signal handlers
        def signal_handler(signum, frame):
            handle_shutdown(orchestrator, monitor)
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start dashboard if requested
        if args.dashboard:
            run_dashboard(app)
        else:
            # Keep the main thread alive
            while True:
                time.sleep(1)
                
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 