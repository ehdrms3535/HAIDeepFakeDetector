
import json
import shutil
from pathlib import Path


def organize_dataset(
    metadata_path: str = "metadata.json",
    source_dir: str = "dataset/video/train_sample_videos",
    output_dir: str = "dataset"
):
    
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    output_path = Path(output_dir)
    real_dir = output_path / "real"
    fake_dir = output_path / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_dir)
    
    real_count = 0
    fake_count = 0
    missing_count = 0
    
    print(f"\nOrganizing files from {source_dir} to {output_dir}...")
    print(f"Total files in metadata: {len(metadata)}")
    
    for filename, info in metadata.items():
        source_file = source_path / filename
        
        if not source_file.exists():
            missing_count += 1
            continue
        
        if info["label"] == "REAL":
            dest_file = real_dir / filename
            shutil.copy2(source_file, dest_file)
            real_count += 1
            if real_count <= 5:
                print(f"[REAL] {filename} -> {dest_file}")
        elif info["label"] == "FAKE":
            dest_file = fake_dir / filename
            shutil.copy2(source_file, dest_file)
            fake_count += 1
            if fake_count <= 5:
                print(f"[FAKE] {filename} -> {dest_file}")
    
    # 결과 출력
    print("\n" + "="*60)
    print("Organization Complete!")
    print("="*60)
    print(f"REAL files copied: {real_count} -> {real_dir}")
    print(f"FAKE files copied: {fake_count} -> {fake_dir}")
    if missing_count > 0:
        print(f"Missing files: {missing_count}")
    print(f"Total processed: {real_count + fake_count}/{len(metadata)}")


def check_dataset_status(output_dir: str = "dataset"):

    output_path = Path(output_dir)
    real_dir = output_path / "real"
    fake_dir = output_path / "fake"
    
    print("\n" + "="*60)
    print("Current Dataset Status")
    print("="*60)
    
    if real_dir.exists():
        real_files = list(real_dir.glob("*.mp4"))
        print(f"REAL files: {len(real_files)} in {real_dir}")
    else:
        print(f"REAL directory does not exist: {real_dir}")
    
    if fake_dir.exists():
        fake_files = list(fake_dir.glob("*.mp4"))
        print(f"FAKE files: {len(fake_files)} in {fake_dir}")
    else:
        print(f"FAKE directory does not exist: {fake_dir}")


if __name__ == "__main__":
    import sys

    check_dataset_status()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        organize_dataset()
        print("\n")
        check_dataset_status()
    else:
        print("\nTo run the organization, use:")
        print("  python organize_dataset.py --run")
