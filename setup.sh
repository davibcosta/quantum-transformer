#!/bin/bash

# Define the environment name and path
ENV_NAME="vit-env"
PARENT_DIR="$(dirname "$PWD")"
ENV_DIR="$PARENT_DIR/$ENV_NAME"

# Function to set up the Jupyter kernel and CUDNN symbolic links
setup_jupyter() {
    echo "Setting up Jupyter kernel and CUDNN symbolic links..."
    python -m ipykernel install --user --name=$ENV_NAME --display-name "$ENV_NAME"
    echo "Jupyter kernel setup complete."
}

# Check if the environment directory already exists
if [ -d "$ENV_DIR" ]; then
    echo "Environment '$ENV_NAME' already exists at $ENV_DIR. Activating it..."
    source "$ENV_DIR/bin/activate"
    setup_jupyter
else
    echo "Environment '$ENV_NAME' does not exist. Creating it now in $PARENT_DIR..."
    
    # Create the virtual environment
    python3.10 -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"
    
    # Install Jupyter kernel
    pip install ipykernel
    setup_jupyter
    
    # Install required packages
    echo "Installing required Python packages..."
    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install netket
    echo "All required packages have been installed."
fi

echo "Setup complete!"
