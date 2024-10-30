# environment_setup.py

"""
YOLO Training System Version 8
Environment Setup Module
- Install dependencies
- Configure GPU settings
- Setup logging
- Mount Google Drive
"""

import os
import yaml
import logging
import torch
import platform
import subprocess
from pathlib import Path
from typing import Dict
from datetime import datetime
from google.colab import drive

# Configure logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   handlers=[
       logging.StreamHandler(),
       logging.FileHandler('environment_setup.log')
   ]
)
logger = logging.getLogger(__name__)

class EnvironmentSetup:
   def __init__(self, base_path: str = "/content/drive/MyDrive/yolo_training"):
       """
       Initialize environment setup
       
       Args:
           base_path: Base directory for all training related files
       """
       self.base_path = Path(base_path)
       self.config_path = self.base_path / "config"
       self.config_file = self.config_path / "environment_config.yaml"
       
       # Create config directory
       self.config_path.mkdir(parents=True, exist_ok=True)
       
       logger.info("Initializing environment setup...")

   def setup_all(self) -> bool:
       """Run complete environment setup"""
       try:
           # Mount Google Drive
           if not self._mount_drive():
               return False

           # Install dependencies
           if not self._install_dependencies():
               return False

           # Setup GPU
           gpu_config = self._setup_gpu()

           # Save configuration
           self._save_config(gpu_config)

           logger.info("Environment setup completed successfully")
           return True

       except Exception as e:
           logger.error(f"Environment setup failed: {e}")
           return False

   def _mount_drive(self) -> bool:
       """Mount Google Drive"""
       try:
           if not Path('/content/drive').exists():
               logger.info("Mounting Google Drive...")
               drive.mount('/content/drive')
           logger.info("Google Drive mounted successfully")
           return True
       except Exception as e:
           logger.error(f"Failed to mount Google Drive: {e}")
           return False

   def _install_dependencies(self) -> bool:
       """Install required packages"""
       try:
           logger.info("Installing dependencies...")
           packages = [
               "ultralytics",
               "opencv-python-headless",
               "psutil"
           ]
           
           for package in packages:
               logger.info(f"Installing {package}...")
               result = subprocess.run(
                   ["pip", "install", "-q", package],
                   capture_output=True,
                   text=True
               )
               if result.returncode != 0:
                   logger.error(f"Failed to install {package}: {result.stderr}")
                   return False
           
           return True

       except Exception as e:
           logger.error(f"Failed to install dependencies: {e}")
           return False

   def _setup_gpu(self) -> Dict:
       """Setup and configure GPU"""
       gpu_config = {
           'available': False,
           'name': None,
           'compute_capability': None,
           'memory': None,
           'cuda_version': None
       }

       try:
           if not torch.cuda.is_available():
               logger.warning("No GPU available!")
               return gpu_config

           # Get GPU information
           gpu_config['available'] = True
           gpu_config['name'] = torch.cuda.get_device_name(0)
           gpu_config['compute_capability'] = torch.cuda.get_device_capability(0)
           gpu_config['cuda_version'] = torch.version.cuda

           # Get memory information
           memory_info = torch.cuda.mem_get_info()
           gpu_config['memory'] = {
               'total': memory_info[1] / 1e9,
               'free': memory_info[0] / 1e9,
               'used': (memory_info[1] - memory_info[0]) / 1e9
           }

           # Configure for optimal performance
           torch.backends.cudnn.benchmark = True
           torch.backends.cuda.matmul.allow_tf32 = True
           torch.backends.cudnn.allow_tf32 = True

           # Set memory allocator
           os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

           logger.info(f"GPU setup completed: {gpu_config['name']}")
           return gpu_config

       except Exception as e:
           logger.error(f"GPU setup failed: {e}")
           return gpu_config

   def _save_config(self, gpu_config: Dict):
       """Save environment configuration"""
       config = {
           'timestamp': datetime.now().isoformat(),
           'platform': {
               'system': platform.system(),
               'python_version': platform.python_version(),
               'torch_version': torch.__version__
           },
           'gpu': gpu_config,
           'paths': {
               'base': str(self.base_path),
               'config': str(self.config_path)
           }
       }

       # Save configuration
       with open(self.config_file, 'w') as f:
           yaml.dump(config, f)
       logger.info(f"Configuration saved to {self.config_file}")

   def verify_setup(self) -> bool:
       """Verify environment setup"""
       try:
           # Check CUDA availability
           if not torch.cuda.is_available():
               logger.error("CUDA is not available")
               return False

           # Verify ultralytics installation
           try:
               import ultralytics
               logger.info(f"Ultralytics version: {ultralytics.__version__}")
           except ImportError:
               logger.error("Ultralytics not installed properly")
               return False

           # Check GPU memory
           memory_info = torch.cuda.mem_get_info()
           free_memory = memory_info[0] / 1e9
           if free_memory < 3.0:  # Minimum 3GB required
               logger.warning(f"Low GPU memory: {free_memory:.1f}GB free")

           # Verify config file
           if not self.config_file.exists():
               logger.error("Configuration file not found")
               return False

           logger.info("Environment verification completed successfully")
           return True

       except Exception as e:
           logger.error(f"Environment verification failed: {e}")
           return False

def main():
   """Main execution function"""
   setup = EnvironmentSetup()
   
   if setup.setup_all():
       if setup.verify_setup():
           logger.info("Environment is ready for training")
       else:
           logger.error("Environment verification failed")
   else:
       logger.error("Environment setup failed")

if __name__ == "__main__":
   main()
