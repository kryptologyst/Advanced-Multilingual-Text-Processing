#!/usr/bin/env python3
"""
Run script for Multilingual Text Processing
"""

import sys
import os
import argparse
from pathlib import Path

def run_command_line():
    """Run the command line interface"""
    print("🚀 Starting Multilingual Text Processing (CLI)")
    try:
        from multilingual_processor import main
        main()
    except ImportError as e:
        print(f"❌ Error importing main module: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

def run_streamlit():
    """Run the Streamlit web interface"""
    print("🌐 Starting Multilingual Text Processing (Web UI)")
    try:
        import subprocess
        subprocess.run(["streamlit", "run", "streamlit_app.py"])
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

def run_tests():
    """Run the test suite"""
    print("🧪 Running test suite")
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "pytest", "test_multilingual.py", "-v"])
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)

def run_setup():
    """Run the setup script"""
    print("⚙️ Running setup")
    try:
        import subprocess
        subprocess.run([sys.executable, "setup.py"])
    except Exception as e:
        print(f"❌ Error running setup: {e}")
        sys.exit(1)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Multilingual Text Processing Runner")
    parser.add_argument(
        "mode",
        choices=["cli", "web", "test", "setup"],
        help="Mode to run: cli (command line), web (Streamlit), test (pytest), setup (installation)"
    )
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if args.mode == "cli":
        run_command_line()
    elif args.mode == "web":
        run_streamlit()
    elif args.mode == "test":
        run_tests()
    elif args.mode == "setup":
        run_setup()

if __name__ == "__main__":
    main()
