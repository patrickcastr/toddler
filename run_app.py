import os
import subprocess
import sys
import importlib.util

if __name__ == "__main__":
    try:
        # Ensure streamlit is installed
        try:
            import streamlit
        except ImportError:
            print("Streamlit not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            print("Streamlit installed successfully.")
        
        # Launch the app
        frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", frontend_path])
        # Removed the standalone print statement here
    except Exception as e:
        print(f"Error launching app: {str(e)}")
        input("Press Enter to exit...")
        sys.exit(1)