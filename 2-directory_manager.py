# directory_manager.py

"""
YOLO Training System Version 8
Directory Management Module
- Create directory structure
- Manage working directories
- Setup backup locations
"""

import os
import shutil
import yaml
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Configure logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   handlers=[
       logging.StreamHandler(),
       logging.FileHandler('directory_setup.log')
   ]
)
logger = logging.getLogger(__name__)

class DirectoryManager:
   def __init__(self, base_path: str = "/content/drive/MyDrive/yolo_training"):
       """
       Initialize directory manager
       
       Args:
           base_path: Base directory for all training files
       """
       self.base_path = Path(base_path)
       self.structure_file = self.base_path / "config/directory_structure.yaml"
       
       # Define directory structure
       self.directories = {
           # Main directories
           'base': self.base_path,
           'config': self.base_path / "config",
           'dataset': self.base_path / "dataset",
           'models': self.base_path / "models",
           'results': self.base_path / "results",
           'logs': self.base_path / "logs",
           
           # Dataset subdirectories
           'images': self.base_path / "dataset/images",
           'labels': self.base_path / "dataset/labels",
           'splits': self.base_path / "dataset/splits",
           
           # Model subdirectories
           'weights': self.base_path / "models/weights",
           'checkpoints': self.base_path / "models/checkpoints",
           'checkpoints_backup': self.base_path / "models/checkpoints_backup",
           
           # Results subdirectories
           'evaluation': self.base_path / "results/evaluation",
           'visualization': self.base_path / "results/visualization",
           
           # Working directories (local)
           'working': Path("/content/yolo_working"),
           'temp': Path("/content/yolo_temp"),
           'cache': Path("/content/yolo_cache")
       }
       
       # Directories that need cleanup
       self.temp_directories = ['working', 'temp', 'cache']

   def setup_all(self) -> bool:
       """Create complete directory structure"""
       try:
           logger.info("Setting up directory structure...")
           
           # Create all directories
           for name, path in self.directories.items():
               path.mkdir(parents=True, exist_ok=True)
               logger.info(f"Created directory: {name} at {path}")
           
           # Save directory structure
           self._save_structure()
           
           # Verify structure
           if self.verify_structure():
               logger.info("Directory structure created successfully")
               return True
           return False
           
       except Exception as e:
           logger.error(f"Failed to setup directories: {e}")
           return False

   def _save_structure(self):
       """Save directory structure to YAML"""
       try:
           # Convert Path objects to strings
           structure_dict = {k: str(v) for k, v in self.directories.items()}
           
           # Create config directory if needed
           self.directories['config'].mkdir(parents=True, exist_ok=True)
           
           # Save structure
           with open(self.structure_file, 'w') as f:
               yaml.dump(structure_dict, f)
           logger.info(f"Directory structure saved to {self.structure_file}")
           
       except Exception as e:
           logger.error(f"Failed to save directory structure: {e}")

   def verify_structure(self) -> bool:
       """Verify all directories exist and are accessible"""
       try:
           for name, path in self.directories.items():
               if not path.exists():
                   logger.error(f"Directory missing: {name} at {path}")
                   return False
               
               # Test write access
               test_file = path / ".test_write"
               try:
                   test_file.touch()
                   test_file.unlink()
               except Exception as e:
                   logger.error(f"No write access to {name} at {path}: {e}")
                   return False
           
           return True
           
       except Exception as e:
           logger.error(f"Directory verification failed: {e}")
           return False

   def cleanup_temp_directories(self):
       """Clean up temporary directories"""
       try:
           for dir_name in self.temp_directories:
               path = self.directories[dir_name]
               if path.exists():
                   shutil.rmtree(path)
                   path.mkdir(parents=True, exist_ok=True)
                   logger.info(f"Cleaned up {dir_name} directory")
                   
       except Exception as e:
           logger.error(f"Failed to cleanup temporary directories: {e}")

   def backup_important_files(self):
       """Backup important files"""
       try:
           # Define important directories to backup
           backup_dirs = ['weights', 'checkpoints', 'config']
           
           for dir_name in backup_dirs:
               source = self.directories[dir_name]
               if source.exists():
                   # Create backup with timestamp
                   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                   backup_path = self.directories['checkpoints_backup'] / f"{dir_name}_{timestamp}"
                   shutil.copytree(source, backup_path, dirs_exist_ok=True)
                   logger.info(f"Backed up {dir_name} to {backup_path}")
                   
       except Exception as e:
           logger.error(f"Failed to create backups: {e}")

   def get_directory_info(self) -> Dict:
       """Get information about directory usage"""
       info = {}
       try:
           for name, path in self.directories.items():
               if path.exists():
                   # Get directory size
                   total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                   # Get file count
                   file_count = sum(1 for _ in path.rglob('*') if _.is_file())
                   info[name] = {
                       'path': str(path),
                       'size_mb': total_size / (1024 * 1024),
                       'file_count': file_count
                   }
           return info
       except Exception as e:
           logger.error(f"Failed to get directory info: {e}")
           return {}

   def create_working_directory(self) -> Path:
       """Create and prepare working directory for training"""
       try:
           working_dir = self.directories['working']
           
           # Clean existing directory
           if working_dir.exists():
               shutil.rmtree(working_dir)
           working_dir.mkdir(parents=True)
           
           # Create subdirectories
           subdirs = ['dataset', 'weights', 'results']
           for subdir in subdirs:
               (working_dir / subdir).mkdir()
           
           logger.info(f"Prepared working directory at {working_dir}")
           return working_dir
           
       except Exception as e:
           logger.error(f"Failed to create working directory: {e}")
           return None

def main():
   """Main execution function"""
   try:
       # Initialize directory manager
       manager = DirectoryManager()
       
       # Setup directories
       if not manager.setup_all():
           logger.error("Directory setup failed")
           return
       
       # Create working directory
       if not manager.create_working_directory():
           logger.error("Failed to create working directory")
           return
       
       # Print directory information
       dir_info = manager.get_directory_info()
       logger.info("\nDirectory Information:")
       for name, info in dir_info.items():
           logger.info(f"{name}:")
           logger.info(f"  Path: {info['path']}")
           logger.info(f"  Size: {info['size_mb']:.2f}MB")
           logger.info(f"  Files: {info['file_count']}")
       
       logger.info("Directory management setup completed successfully")
       
   except Exception as e:
       logger.error(f"Directory management failed: {e}")

if __name__ == "__main__":
   main()
