# setup_yolo_project.py

"""
YOLO Project Structure Setup Script
- Create project directories
- Setup file structure
- Verify permissions
"""

import os
import yaml
from pathlib import Path
from google.colab import drive
import shutil

class ProjectSetup:
   def __init__(self, base_path: str = "/content/drive/MyDrive/yolo_training"):
       """
       Initialize project setup
       
       Args:
           base_path: Base directory for the project
       """
       self.base_path = Path(base_path)
       
       # Define directory structure
       self.directories = {
           # Main directories
           'base': self.base_path,
           'config': self.base_path / "config",
           'dataset': self.base_path / "dataset",
           'models': self.base_path / "models",
           'results': self.base_path / "results",
           
           # Dataset subdirectories
           'images': self.base_path / "dataset/images",
           'labels': self.base_path / "dataset/labels",
           
           # Model subdirectories
           'weights': self.base_path / "models/weights",
           'checkpoints': self.base_path / "models/checkpoints",
           'checkpoints_backup': self.base_path / "models/checkpoints_backup",
           
           # Results subdirectories
           'analysis': self.base_path / "results/analysis",
           'train': self.base_path / "results/train",
           
           # Working directory
           'working': Path("/content/yolo_working")
       }

   def setup(self):
       """Create project structure"""
       print("Setting up YOLO project structure...")
       
       # Mount Google Drive
       if not Path('/content/drive').exists():
           drive.mount('/content/drive')
           print("Mounted Google Drive")
       
       # Create directories
       for name, path in self.directories.items():
           path.mkdir(parents=True, exist_ok=True)
           print(f"Created directory: {name} at {path}")
       
       # Create initial data.yaml
       self._create_data_yaml()
       
       # Create empty train.txt and val.txt
       self._create_split_files()
       
       # Save directory structure
       self._save_directory_structure()
       
       print("\nProject structure setup completed!")
       self._print_structure()

   def _create_data_yaml(self):
       """Create initial data.yaml file"""
       data_yaml = self.directories['dataset'] / 'data.yaml'
       data_config = {
           'path': str(self.directories['dataset']),
           'train': 'train.txt',
           'val': 'val.txt',
           'nc': 1,  # number of classes
           'names': ['container_number']  # class names
       }
       
       with open(data_yaml, 'w') as f:
           yaml.dump(data_config, f, sort_keys=False)
       print(f"Created data.yaml at {data_yaml}")

   def _create_split_files(self):
       """Create empty train.txt and val.txt"""
       for split in ['train.txt', 'val.txt']:
           split_file = self.directories['dataset'] / split
           split_file.touch()
           print(f"Created empty {split}")

   def _save_directory_structure(self):
       """Save directory structure to YAML"""
       structure_file = self.directories['config'] / 'directory_structure.yaml'
       
       # Convert Path objects to strings
       structure_dict = {k: str(v) for k, v in self.directories.items()}
       
       with open(structure_file, 'w') as f:
           yaml.dump(structure_dict, f)
       print(f"\nSaved directory structure to {structure_file}")

   def _print_structure(self):
       """Print project structure"""
       print("\nProject Structure:")
       print(f"Base directory: {self.base_path}")
       print("\nDirectories created:")
       for name, path in self.directories.items():
           print(f"- {name}: {path}")

   def verify_permissions(self):
       """Verify write permissions"""
       print("\nVerifying write permissions...")
       for name, path in self.directories.items():
           try:
               test_file = path / '.test_write'
               test_file.touch()
               test_file.unlink()
               print(f"✓ {name}: Write permission OK")
           except Exception as e:
               print(f"✗ {name}: No write permission - {e}")

def main():
   """Main execution function"""
   try:
       # Initialize and run setup
       setup = ProjectSetup()
       setup.setup()
       
       # Verify permissions
       setup.verify_permissions()
       
       print("\nSetup completed successfully!")
       
   except Exception as e:
       print(f"\nError during setup: {e}")

if __name__ == "__main__":
   main()
