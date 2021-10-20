wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda_11.4.0_470.42.01_linux.run
sh cuda_11.4.0_470.42.01_linux.run
gunicorn -w 4 -k uvicorn.workers.UvicornWorker --timeout 600 main:app