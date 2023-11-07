ENV_NAME := cdan
PYTHON_VERSION := 3.8
# conda run -n $(ENV_NAME) conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# conda run -n $(ENV_NAME) conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

create_env:
	@echo "Updating conda..."
	conda update -n base -c defaults conda -y
	@echo "Creating environment..."
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION)  -y
	@echo "Installing python dependencies..."
	conda run -n $(ENV_NAME) conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
	conda run -n $(ENV_NAME) pip install -r ./requirements.txt


remove_env:
	@echo "Removing environment..."
	conda env remove --name $(ENV_NAME)
	@echo "Environment removed successfully."
