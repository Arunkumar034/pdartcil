import os
import numpy as np
from PIL import Image
import gzip
import struct
from multiprocessing import Pool, cpu_count
from random import shuffle

# Configuration
INPUT_DIR = '4_Png'
OUTPUT_DIR = '5_Mnist'
JOBS = [['Train', 'train'], ['Test', 'test']]

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)

def load_image_worker(file_info):
    """Worker function to process a single image in parallel"""
    filepath, label = file_info
    try:
        # Open image, convert to grayscale, and get pixel data as a flat list
        with Image.open(filepath) as img:
            # Using getdata() is faster than np.array() for single images
            return list(img.getdata()), label, img.size
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def save_mnist_format(base_name, images, labels, h, w):
    """Saves data directly to compressed .gz IDX format"""
    num_items = len(images)
    
    # Save Labels: Magic 2049 (0x00000801)
    label_path = f"{base_name}-labels-idx1-ubyte.gz"
    with gzip.open(label_path, 'wb') as f:
        f.write(struct.pack('>II', 2049, num_items))
        f.write(np.array(labels, dtype=np.uint8).tobytes())

    # Save Images: Magic 2051 (0x00000803)
    image_path = f"{base_name}-images-idx3-ubyte.gz"
    with gzip.open(image_path, 'wb') as f:
        f.write(struct.pack('>IIII', 2051, num_items, h, w))
        f.write(np.array(images, dtype=np.uint8).tobytes())

if __name__ == '__main__':
    mkdir_p(OUTPUT_DIR)

    # 1. Create Label Mapping
    train_path = os.path.join(INPUT_DIR, 'Train')
    class_names = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    label_mapping = {name: i for i, name in enumerate(class_names)}

    with open(os.path.join(OUTPUT_DIR, 'label_mapping.txt'), 'w') as f:
        for name, label in label_mapping.items():
            f.write(f"{label}: {name}\n")
    print(f"✅ Mapping created for {len(class_names)} classes.")

    # 2. Process Sets
    for folder, out_name in JOBS:
        print(f"\n🚀 Processing {folder} set...")
        file_tasks = []
        
        # Collect all file paths and their labels
        base_folder = os.path.join(INPUT_DIR, folder)
        for class_name in class_names:
            class_dir = os.path.join(base_folder, class_name)
            if not os.path.exists(class_dir): continue
            
            label = label_mapping[class_name]
            for fname in os.listdir(class_dir):
                if fname.endswith(".png"):
                    file_tasks.append((os.path.join(class_dir, fname), label))

        shuffle(file_tasks)
        print(f"Found {len(file_tasks)} images. Using {cpu_count()} CPU cores...")

        # 3. Parallel Execution
        # We use a Pool to process images across all CPU cores
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(load_image_worker, file_tasks)

        # 4. Filter results and separate images/labels
        valid_results = [r for r in results if r is not None]
        images_data = [r[0] for r in valid_results]
        labels_data = [r[1] for r in valid_results]
        width, height = valid_results[0][2]

        # 5. Save to compressed files
        print(f"Writing to {OUTPUT_DIR}...")
        save_mnist_format(os.path.join(OUTPUT_DIR, out_name), images_data, labels_data, height, width)
        print(f"✅ Finished {folder} set.")

    print("\n🎉 All tasks complete!")