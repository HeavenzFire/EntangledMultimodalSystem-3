#!/usr/bin/env python3
"""
Global Unity Pulse Runner - A command-line interface for running the Global Unity Pulse visualization.

This script provides a simple interface for running the Global Unity Pulse visualization,
allowing users to specify the number of participants and other parameters.
"""

import argparse
import time
import os
import sys
import logging
from pathlib import Path

# Import the Global Unity Pulse
from global_unity_pulse import GlobalUnityPulse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('unity_pulse.log')
    ]
)

logger = logging.getLogger("UnityPulseRunner")


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run the Global Unity Pulse visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--participants",
        type=int,
        default=1,
        help="The number of participants in the visualization"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="The directory to save output files"
    )
    
    parser.add_argument(
        "--save-state",
        action="store_true",
        help="Save the visualization state to a file"
    )
    
    parser.add_argument(
        "--load-state",
        type=str,
        help="Load the visualization state from a file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def create_output_directory(output_dir):
    """
    Create the output directory if it doesn't exist.
    
    Args:
        output_dir: The output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")


def display_phase_info(phase, elapsed_time, total_time):
    """
    Display information about the current phase.
    
    Args:
        phase: The current phase
        elapsed_time: The elapsed time in minutes
        total_time: The total time in minutes
    """
    print("\n" + "=" * 80)
    print(f"Phase: {phase.name}")
    print(f"Time: {elapsed_time:.1f} / {total_time:.1f} minutes")
    print("-" * 80)
    print(f"Description: {phase.description}")
    print(f"Frequency: {phase.frequency} Hz")
    print(f"Visualization: {phase.visualization_type} ({phase.energy_color})")
    print("-" * 80)
    print("Instructions:")
    for i, instruction in enumerate(phase.instructions, 1):
        print(f"{i}. {instruction}")
    print("=" * 80 + "\n")


def main():
    """Main function to run the Global Unity Pulse visualization."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    create_output_directory(args.output_dir)
    
    # Initialize the Global Unity Pulse
    pulse = GlobalUnityPulse()
    
    # Load state if specified
    if args.load_state:
        try:
            pulse.load_visualization_state(args.load_state)
            logger.info(f"Loaded visualization state from {args.load_state}")
        except Exception as e:
            logger.error(f"Failed to load visualization state: {e}")
            return 1
    
    # Start the visualization
    pulse.start_visualization(participants=args.participants)
    
    # Calculate total duration
    total_duration = sum(phase.duration_minutes for phase in pulse.phases)
    
    # Print initial information
    print("\n" + "=" * 80)
    print("Global Unity Pulse Visualization")
    print("=" * 80)
    print(f"Participants: {args.participants}")
    print(f"Total Duration: {total_duration} minutes")
    print("=" * 80 + "\n")
    
    # Run the visualization
    try:
        while pulse.is_running:
            current_phase = pulse.get_current_phase()
            progress = pulse.get_progress() * 100
            elapsed_time = pulse.get_elapsed_time()
            remaining_time = pulse.get_remaining_time()
            global_field_strength = pulse.get_global_field_strength() * 100
            
            # Clear the screen and display progress
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("\n" + "=" * 80)
            print("Global Unity Pulse Visualization")
            print("=" * 80)
            print(f"Progress: {progress:.1f}%")
            print(f"Elapsed Time: {elapsed_time:.1f} minutes")
            print(f"Remaining Time: {remaining_time:.1f} minutes")
            print(f"Global Field Strength: {global_field_strength:.1f}%")
            print("=" * 80)
            
            # Display phase information if available
            if current_phase:
                display_phase_info(current_phase, elapsed_time, total_duration)
            
            # Save state if requested
            if args.save_state and int(elapsed_time) % 5 == 0 and int(elapsed_time) > 0:
                state_file = Path(args.output_dir) / f"unity_pulse_state_{int(elapsed_time)}.json"
                pulse.save_visualization_state(state_file)
                logger.info(f"Saved visualization state to {state_file}")
            
            # Sleep for a short time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping visualization...")
        pulse.stop_visualization()
    
    # Print completion message
    print("\n" + "=" * 80)
    print("Global Unity Pulse Visualization Completed")
    print("=" * 80)
    print(f"Total Participants: {pulse.participants}")
    print(f"Final Global Field Strength: {pulse.get_global_field_strength() * 100:.1f}%")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 