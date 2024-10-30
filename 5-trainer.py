# trainer.py

"""
YOLO Training System Version 8
Training Module
- Enhanced training process
- Integrated checkpoint management
- Auto-resume capabilities
- Resource monitoring
"""

import os
import gc
import yaml
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from checkpoint_manager import CheckpointManager

# Configure logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   handlers=[
       logging.StreamHandler(),
       logging.FileHandler('training.log')
   ]
)
logger = logging.getLogger(__name__)

class YOLOTrainer:
   def __init__(self, base_path: str = "/content/drive/MyDrive/yolo_training"):
       """
       Initialize YOLO trainer
       
       Args:
           base_path: Base directory for training
       """
       self.base_path = Path(base_path)
       self.load_configs()
       
       # Initialize checkpoint manager
       self.checkpoint_manager = CheckpointManager(self.base_path)
       
       # Disable wandb
       os.environ['WANDB_DISABLED'] = 'true'
       os.environ['WANDB_MODE'] = 'offline'
       
       logger.info("Initialized YOLO trainer")

   def load_configs(self):
       """Load configurations from previous steps"""
       try:
           # Load directory structure
           with open(self.base_path / "config/directory_structure.yaml") as f:
               self.paths = {k: Path(v) for k, v in yaml.safe_load(f).items()}
           
           # Load dataset analysis
           with open(self.paths['analysis'] / "dataset_analysis.yaml") as f:
               self.dataset_info = yaml.safe_load(f)
           
           # Load environment config
           with open(self.base_path / "config/environment_config.yaml") as f:
               self.env_config = yaml.safe_load(f)
           
           logger.info("Loaded all configurations successfully")
           
       except Exception as e:
           logger.error(f"Failed to load configurations: {e}")
           raise

   def setup_training_config(self) -> Dict:
       """Create training configuration based on analysis"""
       try:
           # Get image size stats from dataset analysis
           img_stats = self.dataset_info['details']['images']['statistics']['sizes']
           
           # Calculate optimal batch size based on GPU memory
           gpu_mem = self.env_config['gpu']['memory']['total']
           batch_size = self._calculate_batch_size(gpu_mem)
           
           config = {
               'data': str(self.paths['dataset'] / 'data.yaml'),
               'epochs': 100,
               'imgsz': 640,  # Standard YOLO size
               'batch': batch_size,
               'device': 0,
               'workers': min(8, os.cpu_count() or 1),
               
               # Training parameters
               'patience': 50,
               'save_period': 10,
               'exist_ok': True,
               'pretrained': True,
               'amp': True,  # Mixed precision training
               'verbose': True,
               
               # Project settings
               'project': str(self.paths['results']),
               'name': 'train',
               'save': True,
               
               # Optimization
               'optimizer': 'Adam',
               'lr0': 0.01,
               'weight_decay': 0.0005,
               'warmup_epochs': 3,
               
               # Augmentation (adjusted based on dataset size)
               'hsv_h': 0.015,
               'hsv_s': 0.7,
               'hsv_v': 0.4,
               'degrees': 10,
               'translate': 0.1,
               'scale': 0.5,
               'fliplr': 0.5,
               'mosaic': 0.5,
               'mixup': 0.0
           }
           
           logger.info("Created training configuration")
           return config
           
       except Exception as e:
           logger.error(f"Failed to setup training config: {e}")
           raise

   def _calculate_batch_size(self, gpu_memory: float) -> int:
       """Calculate optimal batch size based on available GPU memory"""
       try:
           # Conservative calculation for T4 GPU
           if gpu_memory >= 15:  # 15GB+
               return 16
           elif gpu_memory >= 12:  # 12-15GB
               return 12
           elif gpu_memory >= 8:  # 8-12GB
               return 8
           else:
               return 4
               
       except Exception as e:
           logger.error(f"Batch size calculation failed: {e}")
           return 8  # Safe default

   def prepare_training(self) -> bool:
       """Prepare for training"""
       try:
           # Clear GPU memory
           torch.cuda.empty_cache()
           gc.collect()
           
           # Create data.yaml
           data_config = {
               'path': str(self.paths['dataset']),
               'train': str(self.paths['dataset'] / 'train.txt'),
               'val': str(self.paths['dataset'] / 'val.txt'),
               'nc': len(self.dataset_info['details']['labels']['classes']),
               'names': [f'class_{i}' for i in range(len(self.dataset_info['details']['labels']['classes']))]
           }
           
           with open(self.paths['dataset'] / 'data.yaml', 'w') as f:
               yaml.dump(data_config, f)
           
           logger.info("Training preparation completed")
           return True
           
       except Exception as e:
           logger.error(f"Training preparation failed: {e}")
           return False

   def train(self, resume: bool = False) -> Optional[Path]:
       """
       Train YOLO model
       
       Args:
           resume: Whether to resume from checkpoint
       """
       try:
           from ultralytics import YOLO
           
           # Prepare training
           if not self.prepare_training():
               raise RuntimeError("Training preparation failed")
           
           # Setup model
           if resume and (checkpoint := self.checkpoint_manager.find_latest_checkpoint()):
               logger.info(f"Resuming from checkpoint: {checkpoint}")
               model = YOLO(checkpoint)
           else:
               logger.info("Starting new training with YOLOv8m")
               model = YOLO('yolov8m.pt')
           
           # Get training configuration
           self.train_config = self.setup_training_config()
           
           # Start training
           logger.info("\nStarting training...")
           best_metric = 0
           try:
               results = model.train(**self.train_config)
               
               # Save final model
               timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
               final_model = self.paths['weights'] / f'best_{timestamp}.pt'
               
               if hasattr(results, 'best_metric'):
                   best_metric = results.best_metric
               
               # Save checkpoints
               self.checkpoint_manager.save_checkpoint(
                   model=model,
                   epoch=-1,  # Final epoch
                   metric=best_metric,
                   is_best=True
               )
               
               logger.info(f"\nTraining completed successfully")
               logger.info(f"Best model saved at: {final_model}")
               logger.info(f"Best metric: {best_metric:.4f}")
               
               return final_model
               
           except KeyboardInterrupt:
               logger.warning("\nTraining interrupted by user")
               self.checkpoint_manager.save_emergency_checkpoint(model)
               
           except Exception as e:
               logger.error(f"Training error: {e}")
               self.checkpoint_manager.save_emergency_checkpoint(model)
               raise e
           
       except Exception as e:
           logger.error(f"Training failed: {e}")
           return None
           
       finally:
           # Cleanup
           self.cleanup()

   def cleanup(self):
       """Cleanup after training"""
       try:
           # Clear GPU memory
           torch.cuda.empty_cache()
           gc.collect()
           
           # Cleanup checkpoint manager
           self.checkpoint_manager.cleanup()
           
           logger.info("Cleanup completed")
           
       except Exception as e:
           logger.error(f"Cleanup failed: {e}")

def main():
   """Main execution function"""
   try:
       trainer = YOLOTrainer()
       
       # Check for existing training
       if trainer.checkpoint_manager.find_latest_checkpoint():
           user_input = input("\nFound existing checkpoint. Resume training? (y/n): ")
           if user_input.lower() == 'y':
               trainer.train(resume=True)
               return
       
       # Start new training
       trainer.train()
       
   except KeyboardInterrupt:
       logger.warning("\nTraining interrupted by user")
   except Exception as e:
       logger.error(f"Training failed: {e}")
   finally:
       logger.info("Training session ended")

if __name__ == "__main__":
   main()
