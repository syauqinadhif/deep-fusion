#!/usr/bin/env python3
"""
KITTI Dataset Downloader for macOS
Run this script to download KITTI 3D Object Detection dataset.
"""

import os
import urllib.request
from pathlib import Path
from tqdm import tqdm

# KITTI dataset URLs
DOWNLOADS = {
    'velodyne': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip',
    'image_2': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip',
    'calib': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip',
    'label_2': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip',
}

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download file with progress bar"""
    filename = os.path.basename(url)
    print(f"\nDownloading {filename}...")

    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=output_path,
                                   reporthook=t.update_to)

def main():
    """Main download function"""
    # Create dataset directory
    dataset_dir = Path.home() / 'Datasets' / 'KITTI'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(dataset_dir)
    print(f"Downloading to: {dataset_dir}")

    print("\n" + "="*60)
    print("KITTI Dataset Downloader")
    print("="*60)
    print("\nThis will download approximately 25 GB of data.")
    print("Make sure you have enough storage space.\n")

    input("Press Enter to start download...")

    # Download each file
    for name, url in DOWNLOADS.items():
        output_path = dataset_dir / f"{name}.zip"

        if output_path.exists():
            print(f"\n✓ {name}.zip already exists, skipping...")
            continue

        try:
            download_url(url, output_path)
            print(f"✓ Downloaded {name}.zip")
        except Exception as e:
            print(f"✗ Failed to download {name}.zip: {e}")

    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Unzip the downloaded files")
    print("2. Organize the KITTI directory structure")
    print("\nSee M4_SETUP.md for detailed instructions")

if __name__ == '__main__':
    main()
