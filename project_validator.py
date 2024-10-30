# project_validator.py

"""
YOLO Training System Version 8
Project Validation Module
- Check system requirements
- Validate dataset structure
- Verify GPU availability
- Check directory permissions
"""

import os
import sys
import torch
import yaml
import cv2
import psutil
from pathlib import Path
from typing import Dict, List

class ProjectValidator:
    def __init__(self):
        self.validation_results = {
            'system': {'status': 'not_checked', 'details': {}},
            'gpu': {'status': 'not_checked', 'details': {}},
            'dataset': {'status': 'not_checked', 'details': {}},
            'storage': {'status': 'not_checked', 'details': {}},
            'permissions': {'status': 'not_checked', 'details': {}}
        }

    def validate_all(self) -> Dict:
        """Run all validation checks"""
        self.check_system_requirements()
        self.check_gpu_availability()
        self.check_dataset_structure()
        self.check_storage_space()
        self.check_permissions()
        return self.validation_results

    def check_system_requirements(self):
        """Check Python version and required packages"""
        try:
            requirements = {
                'python': '>=3.7.0',
                'packages': {
                    'torch': '>=2.0.0',
                    'ultralytics': '>=8.0.0',
                    'opencv-python-headless': '>=4.7.0',
                    'pyyaml': '>=6.0'
                }
            }

            details = {
                'python_version': sys.version,
                'packages': {},
                'missing_packages': []
            }

            # Check packages
            for package, required_version in requirements['packages'].items():
                try:
                    if package == 'torch':
                        details['packages'][package] = torch.__version__
                    elif package == 'opencv-python-headless':
                        details['packages'][package] = cv2.__version__
                    elif package == 'pyyaml':
                        details['packages'][package] = yaml.__version__
                    else:
                        module = __import__(package)
                        details['packages'][package] = module.__version__
                except ImportError:
                    details['missing_packages'].append(package)

            self.validation_results['system'] = {
                'status': 'passed' if not details['missing_packages'] else 'failed',
                'details': details
            }

        except Exception as e:
            self.validation_results['system'] = {
                'status': 'error',
                'details': {'error': str(e)}
            }

    def check_gpu_availability(self):
        """Check GPU availability and specifications"""
        try:
            details = {
                'available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count(),
                'device_name': None,
                'memory': None,
                'cuda_version': torch.version.cuda
            }

            if details['available']:
                details['device_name'] = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory
                details['memory'] = f"{memory/1e9:.1f}GB"

            self.validation_results['gpu'] = {
                'status': 'passed' if details['available'] else 'failed',
                'details': details
            }

        except Exception as e:
            self.validation_results['gpu'] = {
                'status': 'error',
                'details': {'error': str(e)}
            }

    def check_dataset_structure(self, dataset_path: str = "/content/drive/MyDrive/yolo_training/dataset"):
        """Check dataset structure and files"""
        try:
            dataset_path = Path(dataset_path)
            details = {
                'structure': {
                    'images': {'exists': False, 'count': 0},
                    'labels': {'exists': False, 'count': 0},
                    'data_yaml': {'exists': False},
                    'train_txt': {'exists': False},
                    'val_txt': {'exists': False}
                },
                'sample_check': {'images': [], 'labels': []}
            }

            # Check directories and files
            if (dataset_path / 'images').exists():
                details['structure']['images']['exists'] = True
                details['structure']['images']['count'] = len(list((dataset_path / 'images').glob('*.*')))
                # Sample some images
                for img_path in list((dataset_path / 'images').glob('*.*'))[:3]:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        details['sample_check']['images'].append({
                            'name': img_path.name,
                            'size': f"{img.shape[1]}x{img.shape[0]}"
                        })

            if (dataset_path / 'labels').exists():
                details['structure']['labels']['exists'] = True
                details['structure']['labels']['count'] = len(list((dataset_path / 'labels').glob('*.txt')))
                # Sample some labels
                for label_path in list((dataset_path / 'labels').glob('*.txt'))[:3]:
                    with open(label_path) as f:
                        details['sample_check']['labels'].append({
                            'name': label_path.name,
                            'lines': len(f.readlines())
                        })

            details['structure']['data_yaml']['exists'] = (dataset_path / 'data.yaml').exists()
            details['structure']['train_txt']['exists'] = (dataset_path / 'train.txt').exists()
            details['structure']['val_txt']['exists'] = (dataset_path / 'val.txt').exists()

            # Validate status
            status = all([
                details['structure']['images']['exists'],
                details['structure']['labels']['exists'],
                details['structure']['data_yaml']['exists'],
                details['structure']['train_txt']['exists'],
                details['structure']['val_txt']['exists'],
                details['structure']['images']['count'] > 0,
                details['structure']['labels']['count'] > 0
            ])

            self.validation_results['dataset'] = {
                'status': 'passed' if status else 'failed',
                'details': details
            }

        except Exception as e:
            self.validation_results['dataset'] = {
                'status': 'error',
                'details': {'error': str(e)}
            }

    def check_storage_space(self, required_gb: float = 10.0):
        """Check available storage space"""
        try:
            details = {}
            for path in ['/content', '/content/drive']:
                if os.path.exists(path):
                    total, used, free = shutil.disk_usage(path)
                    details[path] = {
                        'total_gb': total / (1024**3),
                        'free_gb': free / (1024**3),
                        'used_gb': used / (1024**3)
                    }

            # Check if we have enough space
            status = any(info['free_gb'] >= required_gb for info in details.values())
            
            self.validation_results['storage'] = {
                'status': 'passed' if status else 'failed',
                'details': details
            }

        except Exception as e:
            self.validation_results['storage'] = {
                'status': 'error',
                'details': {'error': str(e)}
            }

    def check_permissions(self):
        """Check write permissions in key directories"""
        try:
            test_paths = [
                '/content/drive/MyDrive/yolo_training',
                '/content',
                '/content/drive'
            ]
            
            details = {}
            for path in test_paths:
                if os.path.exists(path):
                    test_file = Path(path) / '.write_test'
                    try:
                        test_file.touch()
                        test_file.unlink()
                        details[path] = {'writable': True}
                    except Exception as e:
                        details[path] = {'writable': False, 'error': str(e)}

            status = any(info.get('writable', False) for info in details.values())
            
            self.validation_results['permissions'] = {
                'status': 'passed' if status else 'failed',
                'details': details
            }

        except Exception as e:
            self.validation_results['permissions'] = {
                'status': 'error',
                'details': {'error': str(e)}
            }

    def get_summary(self) -> str:
        """Generate human-readable summary"""
        summary = ["=== Project Validation Summary ===\n"]
        
        for category, results in self.validation_results.items():
            status = results['status']
            summary.append(f"{category.upper()}: {status}")
            
            if status == 'failed' or status == 'error':
                if isinstance(results['details'], dict):
                    if 'error' in results['details']:
                        summary.append(f"  Error: {results['details']['error']}")
                    else:
                        for key, value in results['details'].items():
                            summary.append(f"  {key}: {value}")
            
            summary.append("")
        
        return "\n".join(summary)

def main():
    """Run validation and print results"""
    validator = ProjectValidator()
    validator.validate_all()
    print(validator.get_summary())
    
    # Save results to file
    with open('validation_results.yaml', 'w') as f:
        yaml.dump(validator.validation_results, f)

if __name__ == "__main__":
    main()
