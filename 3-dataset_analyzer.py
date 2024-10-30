# dataset_analyzer.py

"""
YOLO Training System Version 8
Dataset Analysis Module
- Analyze images and labels
- Validate dataset structure
- Generate dataset statistics
"""

import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

# Configure logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   handlers=[
       logging.StreamHandler(),
       logging.FileHandler('dataset_analysis.log')
   ]
)
logger = logging.getLogger(__name__)

class DatasetAnalyzer:
   def __init__(self, base_path: str = "/content/drive/MyDrive/yolo_training"):
       """
       Initialize dataset analyzer
       
       Args:
           base_path: Base directory containing the dataset
       """
       self.base_path = Path(base_path)
       
       # Load directory structure
       self.load_directory_structure()
       
       # Initialize analysis results
       self.analysis_results = {
           'status': 'not_started',
           'timestamp': None,
           'summary': {},
           'details': {}
       }

   def load_directory_structure(self):
       """Load directory structure from config"""
       try:
           structure_file = self.base_path / "config/directory_structure.yaml"
           with open(structure_file) as f:
               structure = yaml.safe_load(f)
           
           self.paths = {
               'images': Path(structure['images']),
               'labels': Path(structure['labels']),
               'splits': Path(structure['splits']),
               'analysis': Path(structure['results']) / 'analysis'
           }
           
           # Create analysis directory
           self.paths['analysis'].mkdir(parents=True, exist_ok=True)
           
       except Exception as e:
           logger.error(f"Failed to load directory structure: {e}")
           raise

   def analyze_dataset(self, sample_size: int = 100) -> Dict:
       """
       Perform complete dataset analysis
       
       Args:
           sample_size: Number of samples to analyze for detailed checks
       """
       try:
           logger.info("Starting dataset analysis...")
           self.analysis_results['timestamp'] = datetime.now().isoformat()
           
           # Basic dataset structure check
           if not self._verify_dataset_structure():
               return self.analysis_results
           
           # Analyze images
           image_stats = self._analyze_images(sample_size)
           self.analysis_results['details']['images'] = image_stats
           
           # Analyze labels
           label_stats = self._analyze_labels(sample_size)
           self.analysis_results['details']['labels'] = label_stats
           
           # Analyze splits
           split_stats = self._analyze_splits()
           self.analysis_results['details']['splits'] = split_stats
           
           # Generate summary
           self._generate_summary()
           
           # Save analysis results
           self._save_analysis_results()
           
           self.analysis_results['status'] = 'completed'
           logger.info("Dataset analysis completed successfully")
           return self.analysis_results
           
       except Exception as e:
           logger.error(f"Dataset analysis failed: {e}")
           self.analysis_results['status'] = 'failed'
           self.analysis_results['error'] = str(e)
           return self.analysis_results

   def _verify_dataset_structure(self) -> bool:
       """Verify basic dataset structure"""
       try:
           # Check required directories
           for name, path in self.paths.items():
               if not path.exists():
                   logger.error(f"Required directory missing: {name} at {path}")
                   return False
           
           # Check for required split files
           split_files = ['train.txt', 'val.txt']
           for split_file in split_files:
               if not (self.paths['splits'] / split_file).exists():
                   logger.error(f"Required split file missing: {split_file}")
                   return False
           
           return True
           
       except Exception as e:
           logger.error(f"Dataset structure verification failed: {e}")
           return False

   def _analyze_images(self, sample_size: int) -> Dict:
       """Analyze image characteristics"""
       stats = {
           'total_count': 0,
           'sizes': defaultdict(int),
           'formats': defaultdict(int),
           'aspect_ratios': defaultdict(int),
           'corrupted': [],
           'statistics': {
               'sizes': {'min': None, 'max': None, 'mean': None},
               'aspect_ratios': {'min': None, 'max': None, 'mean': None}
           }
       }
       
       try:
           image_files = list(self.paths['images'].glob('*.*'))
           stats['total_count'] = len(image_files)
           
           if stats['total_count'] == 0:
               logger.warning("No images found!")
               return stats
           
           # Analyze sample of images
           sample_size = min(sample_size, len(image_files))
           sampled_files = np.random.choice(image_files, sample_size, False)
           
           sizes = []
           aspect_ratios = []
           
           for img_path in tqdm(sampled_files, desc="Analyzing images"):
               try:
                   # Get image format
                   stats['formats'][img_path.suffix.lower()] += 1
                   
                   # Read and analyze image
                   img = cv2.imread(str(img_path))
                   if img is None:
                       stats['corrupted'].append(str(img_path))
                       continue
                   
                   h, w = img.shape[:2]
                   sizes.append((w, h))
                   stats['sizes'][f"{w}x{h}"] += 1
                   
                   aspect_ratio = round(w / h, 2)
                   aspect_ratios.append(aspect_ratio)
                   stats['aspect_ratios'][str(aspect_ratio)] += 1
                   
               except Exception as e:
                   stats['corrupted'].append(f"{img_path}: {str(e)}")
           
           # Calculate statistics
           if sizes:
               sizes = np.array(sizes)
               stats['statistics']['sizes'] = {
                   'min': sizes.min(axis=0).tolist(),
                   'max': sizes.max(axis=0).tolist(),
                   'mean': sizes.mean(axis=0).tolist()
               }
           
           if aspect_ratios:
               aspect_ratios = np.array(aspect_ratios)
               stats['statistics']['aspect_ratios'] = {
                   'min': float(aspect_ratios.min()),
                   'max': float(aspect_ratios.max()),
                   'mean': float(aspect_ratios.mean())
               }
           
           return stats
           
       except Exception as e:
           logger.error(f"Image analysis failed: {e}")
           return stats

   def _analyze_labels(self, sample_size: int) -> Dict:
       """Analyze label characteristics"""
       stats = {
           'total_count': 0,
           'classes': defaultdict(int),
           'boxes_per_image': [],
           'corrupted': [],
           'statistics': {
               'boxes_per_image': {'min': None, 'max': None, 'mean': None},
               'box_sizes': {'min': None, 'max': None, 'mean': None}
           }
       }
       
       try:
           label_files = list(self.paths['labels'].glob('*.txt'))
           stats['total_count'] = len(label_files)
           
           if stats['total_count'] == 0:
               logger.warning("No labels found!")
               return stats
           
           # Analyze sample of labels
           sample_size = min(sample_size, len(label_files))
           sampled_files = np.random.choice(label_files, sample_size, False)
           
           box_sizes = []
           
           for label_path in tqdm(sampled_files, desc="Analyzing labels"):
               try:
                   with open(label_path) as f:
                       lines = f.readlines()
                   
                   stats['boxes_per_image'].append(len(lines))
                   
                   for line in lines:
                       try:
                           parts = line.strip().split()
                           if len(parts) != 5:
                               raise ValueError("Invalid format")
                           
                           class_id = int(parts[0])
                           stats['classes'][class_id] += 1
                           
                           # Calculate box size (width * height)
                           w, h = float(parts[3]), float(parts[4])
                           box_sizes.append(w * h)
                           
                       except Exception as e:
                           stats['corrupted'].append(f"{label_path}: {str(e)}")
                           
               except Exception as e:
                   stats['corrupted'].append(f"{label_path}: {str(e)}")
           
           # Calculate statistics
           if stats['boxes_per_image']:
               boxes = np.array(stats['boxes_per_image'])
               stats['statistics']['boxes_per_image'] = {
                   'min': int(boxes.min()),
                   'max': int(boxes.max()),
                   'mean': float(boxes.mean())
               }
           
           if box_sizes:
               box_sizes = np.array(box_sizes)
               stats['statistics']['box_sizes'] = {
                   'min': float(box_sizes.min()),
                   'max': float(box_sizes.max()),
                   'mean': float(box_sizes.mean())
               }
           
           return stats
           
       except Exception as e:
           logger.error(f"Label analysis failed: {e}")
           return stats

   def _analyze_splits(self) -> Dict:
       """Analyze dataset splits"""
       stats = {
           'train': {'count': 0, 'missing_images': [], 'missing_labels': []},
           'val': {'count': 0, 'missing_images': [], 'missing_labels': []}
       }
       
       try:
           for split in ['train', 'val']:
               split_file = self.paths['splits'] / f"{split}.txt"
               if not split_file.exists():
                   continue
               
               with open(split_file) as f:
                   image_paths = [line.strip() for line in f.readlines()]
               
               stats[split]['count'] = len(image_paths)
               
               # Check for missing files
               for img_path in image_paths:
                   img_file = self.paths['images'] / img_path
                   label_file = self.paths['labels'] / f"{img_path.rsplit('.', 1)[0]}.txt"
                   
                   if not img_file.exists():
                       stats[split]['missing_images'].append(str(img_file))
                   if not label_file.exists():
                       stats[split]['missing_labels'].append(str(label_file))
           
           return stats
           
       except Exception as e:
           logger.error(f"Split analysis failed: {e}")
           return stats

   def _generate_summary(self):
       """Generate dataset summary"""
       try:
           details = self.analysis_results['details']
           self.analysis_results['summary'] = {
               'total_images': details['images']['total_count'],
               'total_labels': details['labels']['total_count'],
               'num_classes': len(details['labels']['classes']),
               'train_samples': details['splits']['train']['count'],
               'val_samples': details['splits']['val']['count'],
               'corrupted_images': len(details['images']['corrupted']),
               'corrupted_labels': len(details['labels']['corrupted']),
               'missing_files': {
                   'train': {
                       'images': len(details['splits']['train']['missing_images']),
                       'labels': len(details['splits']['train']['missing_labels'])
                   },
                   'val': {
                       'images': len(details['splits']['val']['missing_images']),
                       'labels': len(details['splits']['val']['missing_labels'])
                   }
               }
           }
       except Exception as e:
           logger.error(f"Failed to generate summary: {e}")

   def _save_analysis_results(self):
       """Save analysis results to file"""
       try:
           output_file = self.paths['analysis'] / 'dataset_analysis.yaml'
           with open(output_file, 'w') as f:
               yaml.dump(self.analysis_results, f)
           logger.info(f"Analysis results saved to {output_file}")
       except Exception as e:
           logger.error(f"Failed to save analysis results: {e}")

def main():
   """Main execution function"""
   try:
       analyzer = DatasetAnalyzer()
       results = analyzer.analyze_dataset()
       
       if results['status'] == 'completed':
           logger.info("\nDataset Analysis Summary:")
           for key, value in results['summary'].items():
               logger.info(f"{key}: {value}")
       else:
           logger.error("Dataset analysis failed")
           
   except Exception as e:
       logger.error(f"Dataset analysis failed: {e}")

if __name__ == "__main__":
   main()
