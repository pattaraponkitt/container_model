# checkpoint_manager.py

"""
YOLO Training System Version 8
Checkpoint Management Module
- Enhanced checkpoint saving and loading
- Backup system
- Checkpoint validation
- Anti-disconnect integration
"""

import os
import shutil
import yaml
import logging
import torch
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from IPython.display import display, Javascript

# Configure logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   handlers=[
       logging.StreamHandler(),
       logging.FileHandler('checkpoint.log')
   ]
)
logger = logging.getLogger(__name__)

class CheckpointManager:
   def __init__(self, base_path: str = "/content/drive/MyDrive/yolo_training"):
       """
       Initialize checkpoint manager
       
       Args:
           base_path: Base directory for training
       """
       self.base_path = Path(base_path)
       self.load_paths()
       
       # Initialize tracking
       self.checkpoint_history = []
       self.last_checkpoint_time = time.time()
       self.checkpoint_interval = 900  # 15 minutes
       self.max_checkpoints = 5  # Maximum number of regular checkpoints to keep
       
       # Setup anti-disconnect
       self._setup_anti_disconnect()
       
       logger.info("Initialized checkpoint management system")

   def load_paths(self):
       """Load directory paths from config"""
       try:
           with open(self.base_path / "config/directory_structure.yaml") as f:
               paths = yaml.safe_load(f)
           
           self.paths = {
               'checkpoints': Path(paths['checkpoints']),
               'backup': Path(paths['checkpoints_backup']),
               'temp': Path(paths['temp']),
               'working': Path(paths['working'])
           }
           
           # Create directories if needed
           for path in self.paths.values():
               path.mkdir(parents=True, exist_ok=True)
               
       except Exception as e:
           logger.error(f"Failed to load paths: {e}")
           raise

   def _setup_anti_disconnect(self):
       """Setup anti-disconnect system for Colab"""
       display(Javascript('''
           function keepAlive() {
               function click() {
                   console.log("Keeping connection alive...");
                   document.querySelector("#top-toolbar > colab-connect-button")
                       .shadowRoot.querySelector("#connect").click();
               }
               
               function checkConnection() {
                   const toolbar = document.querySelector("#top-toolbar");
                   const status = toolbar ? toolbar.getAttribute("connection-status") : "ok";
                   
                   if (status !== "ok") {
                       console.log("Connection lost, reconnecting...");
                       click();
                   }
               }
               
               // Regular connection check
               setInterval(checkConnection, 30000);  // Every 30 seconds
               
               // Periodic activity simulation
               setInterval(() => {
                   document.dispatchEvent(new KeyboardEvent('keydown', {'key': 'Shift'}));
               }, 60000);  // Every minute
           }
           
           keepAlive();
       '''))

   def find_latest_checkpoint(self) -> Optional[Path]:
       """Find most recent valid checkpoint"""
       try:
           # Search in all possible locations
           checkpoint_locations = [
               self.paths['checkpoints'],
               self.paths['backup'],
               self.paths['temp']
           ]
           
           valid_checkpoints = []
           
           for location in checkpoint_locations:
               checkpoints = list(location.glob("*.pt"))
               for checkpoint in checkpoints:
                   if self._validate_checkpoint(checkpoint):
                       valid_checkpoints.append(checkpoint)
           
           if not valid_checkpoints:
               logger.info("No valid checkpoints found")
               return None
           
           # Get most recent valid checkpoint
           latest = max(valid_checkpoints, key=lambda p: p.stat().st_mtime)
           logger.info(f"Found latest valid checkpoint: {latest}")
           return latest
           
       except Exception as e:
           logger.error(f"Error finding latest checkpoint: {e}")
           return None

   def _validate_checkpoint(self, checkpoint_path: Path) -> bool:
       """Validate checkpoint file"""
       try:
           if not checkpoint_path.exists():
               return False
               
           # Check file size
           if checkpoint_path.stat().st_size < 1000:  # Minimum size check
               logger.warning(f"Checkpoint too small: {checkpoint_path}")
               return False
           
           # Try loading checkpoint
           try:
               # Load just the metadata without full model
               checkpoint = torch.load(checkpoint_path, map_location='cpu')
               if not isinstance(checkpoint, dict):
                   raise ValueError("Invalid checkpoint format")
           except Exception as e:
               logger.warning(f"Invalid checkpoint {checkpoint_path}: {e}")
               return False
           
           return True
           
       except Exception as e:
           logger.error(f"Checkpoint validation failed: {e}")
           return False

   def save_checkpoint(self, model, epoch: int, optimizer=None, 
                      is_best: bool = False, metric: float = None) -> bool:
       """
       Save checkpoint with backup
       
       Args:
           model: Training model
           epoch: Current epoch
           optimizer: Optimizer state (optional)
           is_best: Whether this is the best model so far
           metric: Performance metric (optional)
       """
       try:
           timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           
           # Prepare checkpoint data
           checkpoint_data = {
               'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'timestamp': timestamp
           }
           
           if optimizer:
               checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
           if metric is not None:
               checkpoint_data['metric'] = metric
           
           # Create checkpoint names
           checkpoint_name = f"checkpoint_epoch{epoch}_{timestamp}.pt"
           backup_name = f"backup_epoch{epoch}_{timestamp}.pt"
           
           # Primary save
           primary_path = self.paths['checkpoints'] / checkpoint_name
           torch.save(checkpoint_data, primary_path)
           
           # Backup save
           backup_path = self.paths['backup'] / backup_name
           torch.save(checkpoint_data, backup_path)
           
           # Save best model separately
           if is_best:
               best_path = self.paths['checkpoints'] / f"best_{timestamp}.pt"
               best_backup_path = self.paths['backup'] / f"best_{timestamp}.pt"
               torch.save(checkpoint_data, best_path)
               torch.save(checkpoint_data, best_backup_path)
           
           # Update history and clean old checkpoints
           self._update_checkpoint_history(primary_path)
           
           logger.info(f"Saved checkpoint at epoch {epoch}")
           return True
           
       except Exception as e:
           logger.error(f"Failed to save checkpoint: {e}")
           return False

   def _update_checkpoint_history(self, new_checkpoint: Path):
       """Update checkpoint history and remove old ones"""
       try:
           self.checkpoint_history.append(new_checkpoint)
           
           # Keep only recent regular checkpoints (keep all best models)
           regular_checkpoints = [cp for cp in self.checkpoint_history 
                                if 'best' not in cp.name]
           
           if len(regular_checkpoints) > self.max_checkpoints:
               # Remove oldest checkpoints
               for old_cp in regular_checkpoints[:-self.max_checkpoints]:
                   try:
                       if old_cp.exists():
                           old_cp.unlink()
                       # Also remove corresponding backup
                       backup_cp = self.paths['backup'] / old_cp.name
                       if backup_cp.exists():
                           backup_cp.unlink()
                   except Exception as e:
                       logger.warning(f"Failed to remove old checkpoint {old_cp}: {e}")
               
               # Update history
               self.checkpoint_history = [cp for cp in self.checkpoint_history 
                                        if cp.exists()]
               
       except Exception as e:
           logger.error(f"Failed to update checkpoint history: {e}")

   def save_emergency_checkpoint(self, model) -> bool:
       """Save emergency checkpoint when interrupted"""
       try:
           timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           emergency_name = f"emergency_{timestamp}.pt"
           
           # Save to multiple locations for safety
           save_locations = [
               self.paths['temp'] / emergency_name,
               self.paths['checkpoints'] / emergency_name,
               self.paths['backup'] / emergency_name
           ]
           
           saved = False
           for location in save_locations:
               try:
                   torch.save(model.state_dict(), location)
                   logger.info(f"Saved emergency checkpoint to {location}")
                   saved = True
               except Exception as e:
                   logger.warning(f"Failed to save emergency checkpoint to {location}: {e}")
           
           return saved
           
       except Exception as e:
           logger.error(f"Failed to save emergency checkpoint: {e}")
           return False

   def should_save_checkpoint(self) -> bool:
       """Check if it's time for periodic checkpoint"""
       current_time = time.time()
       if current_time - self.last_checkpoint_time >= self.checkpoint_interval:
           self.last_checkpoint_time = current_time
           return True
       return False

   def cleanup(self):
       """Cleanup temporary checkpoint files"""
       try:
           # Clean temp directory
           if self.paths['temp'].exists():
               shutil.rmtree(self.paths['temp'])
               self.paths['temp'].mkdir()
           
           logger.info("Cleaned up temporary checkpoint files")
           
       except Exception as e:
           logger.error(f"Cleanup failed: {e}")

def main():
   """Test checkpoint management system"""
   try:
       manager = CheckpointManager()
       
       # Test checkpoint finding
       latest = manager.find_latest_checkpoint()
       if latest:
           logger.info(f"Found latest checkpoint: {latest}")
       else:
           logger.info("No existing checkpoints found")
       
       logger.info("Checkpoint management system test completed")
       
   except Exception as e:
       logger.error(f"Checkpoint management test failed: {e}")

if __name__ == "__main__":
   main()
