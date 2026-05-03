import os
from ultralytics import YOLO

if __name__ == '__main__':
    # Initialize YOLO11 Nano model
    model = YOLO('yolo11n.pt') 

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(current_dir, 'data.yaml')
    project_path = os.path.join(project_root, 'runs', 'detect')

    print("Starting optimized training for RTX 5060 (8GB VRAM)...")
    results = model.train(
        data=data_path,
        epochs=300,
        imgsz=640,
        
        # --- 8GB VRAM Optimization (Critical Section) ---
        batch=8,               # Keep batch small. Exceeding 8GB physical VRAM will trigger system RAM swap, severely bottlenecking speed.
        workers=4,             # Reduce DataLoader workers. High workers consume excessive system RAM on Windows.
        amp=True,              # Enable Automatic Mixed Precision (FP16). Saves VRAM and accelerates training on RTX 50-series GPUs.
        cache=False,           # Do NOT cache images to RAM to prevent memory overflow conflicts with shared VRAM.
        device=0,              # Force use of the dedicated RTX 5060 GPU.
        
        # --- Strategy & Callbacks ---
        save_period=60,
        dropout=0.1,
        patience=0,
        
        # --- Optimal Augmentations for Edge-AI ---
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        mosaic=1.0,
        close_mosaic=30,
        mixup=0.1,
        
        project=project_path,
        name='train_yolo11s_Pi5_Optimized' 
    )

    # --- Deployment Optimization for Raspberry Pi 5 ---
    # The Pi 5 CPU struggles with native PyTorch (.pt) files. 
    # NCNN is a high-performance neural network inference framework optimized for ARM CPUs.
    print("Training complete. Exporting to NCNN format for Raspberry Pi 5 CPU inference...")
    model.export(format='ncnn', half=True) # half=True uses FP16 precision to double inference speed on Pi 5.