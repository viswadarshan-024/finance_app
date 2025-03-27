# requirements.txt
# langchain==0.1.20
# langchain-groq==0.0.3
# streamlit==1.30.0
# yfinance==0.2.33
# duckduckgo-search==3.9.6
# numpy==1.24.4
# setuptools>=69.2.0

# install_dependencies.py
import subprocess
import sys
import os

def install_dependencies():
    print("Starting dependency installation...")
    
    # Ensure pip is up to date
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    
    # Temporarily disable Windows Defender real-time protection
    try:
        subprocess.run(['powershell', 'Set-MpPreference -DisableRealtimeMonitoring $true'], capture_output=True)
        print("Temporarily disabled Windows Defender real-time protection")
    except Exception as e:
        print(f"Could not disable Windows Defender: {e}")
    
    # Install dependencies with additional flags
    dependencies = [
        'numpy==1.24.4',  # Use an older, more stable version
        'setuptools>=69.2.0',
        'wheel',
        'langchain==0.1.20',
        'langchain-groq==0.0.3',
        'streamlit==1.30.0',
        'yfinance==0.2.33',
        'duckduckgo-search==3.9.6'
    ]
    
    for package in dependencies:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                package, 
                '--no-cache-dir',  # Bypass local cache
                '--no-deps',  # Avoid dependency conflicts
                '--force-reinstall'  # Force reinstallation
            ])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
    
    # Re-enable Windows Defender real-time protection
    try:
        subprocess.run(['powershell', 'Set-MpPreference -DisableRealtimeMonitoring $false'], capture_output=True)
        print("Re-enabled Windows Defender real-time protection")
    except Exception as e:
        print(f"Could not re-enable Windows Defender: {e}")

if __name__ == '__main__':
    install_dependencies()

# Installation Instructions
"""
1. Save this as install_dependencies.py
2. Open PowerShell or Command Prompt as Administrator
3. Navigate to the script directory
4. Run: python install_dependencies.py
5. After successful installation, run your Streamlit app
"""