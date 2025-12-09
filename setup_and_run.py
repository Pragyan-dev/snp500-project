import os
import sys
import subprocess
import importlib
import time

# Set API keys as environment variables
os.environ["FRED_API_KEY"] = "Enter Your Fred api key"
os.environ["REDDIT_CLIENT_SECRET"] ="enter your reddit key"
os.environ["GEMINI_API_KEY"] = "Enter your gemini key"

# For Reddit API, set the client ID
os.environ["REDDIT_CLIENT_ID"] = "Enter your reddit client id"

print("API keys set as environment variables")

def setup_and_run_model():
    """
    Set up the necessary environment for the model and run it
    """
    print("Setting up and running the improved S&P 500 model...")
    
    # Check and install required packages
    required_packages = [
        "numpy",
        "pandas",
        "tensorflow",
        "sklearn",
        "matplotlib",
        "yfinance",
        "requests",
        "bs4",
        "torch",
        "transformers",
        "praw",
        "google-generativeai",
        "fredapi"
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            print(f"[OK] {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Import and run the model
    try:
        import improved_sp500_model
        # Force reload to avoid cached imports
        importlib.reload(improved_sp500_model)
        # Run the model - now returns a dict of models
        models, data = improved_sp500_model.train_and_evaluate_model()
        # The ensemble prediction is already done in train_and_evaluate_model
        # No need to call predict_tomorrow_price separately
        # Ensure that the model runs fully by adding a short delay
        time.sleep(1)
        return True
    except Exception as e:
        print(f"Error running the model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    setup_and_run_model() 
