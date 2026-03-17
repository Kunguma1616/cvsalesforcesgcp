#!/usr/bin/env python3
"""
Dependency Verification Script
Checks that all critical dependencies are installed and importable
"""

import sys
import importlib
from packaging import version

REQUIRED_PACKAGES = {
    'fastapi': '0.100.0',
    'uvicorn': '0.23.0',
    'pydantic': '2.0.0',
    'langchain': '0.1.0',
    'langchain_core': '0.1.0',
    'langchain_groq': '0.0.1',
    'groq': '0.4.0',
    'pdfminer': '20240706',
    'reportlab': '4.0.0',
    'dotenv': '1.0.0',
    'azure': '1.0.0',
}

def check_import(package_name, import_name=None):
    """Check if a package can be imported."""
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    try:
        module = importlib.import_module(import_name)
        return True, getattr(module, '__version__', 'unknown')
    except ImportError as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("Dependency Verification")
    print("=" * 60)
    print()
    
    failed = []
    
    packages_to_check = [
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('python-multipart', 'multipart'),
        ('pydantic', 'pydantic'),
        ('langchain', 'langchain'),
        ('langchain-core', 'langchain_core'),
        ('langchain-groq', 'langchain_groq'),
        ('groq', 'groq'),
        ('pdfminer.six', 'pdfminer'),
        ('reportlab', 'reportlab'),
        ('matplotlib', 'matplotlib'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('python-dotenv', 'dotenv'),
        ('azure-storage-blob', 'azure'),
        ('simple-salesforce', 'salesforce'),
    ]
    
    for package, import_name in packages_to_check:
        success, version_info = check_import(import_name)
        status = "✓" if success else "✗"
        print(f"{status} {package:25} {version_info}")
        
        if not success:
            failed.append(package)
    
    print()
    print("=" * 60)
    
    if failed:
        print(f"FAILED: {len(failed)} package(s) missing!")
        print()
        print("To fix, run:")
        print(f"  pip install {' '.join(failed)}")
        print()
        return 1
    else:
        print("SUCCESS: All dependencies installed! ✓")
        print()
        return 0

if __name__ == '__main__':
    sys.exit(main())
