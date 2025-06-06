# 🚗 Carla2Cosmos

## ▶️ Recording a Clip

Stream and record synchronous CARLA sessions at **30 Hz**, saving only the essential data needed to convert each clip into the **RDS-HQ** format (used by Cosmos-AV). To record the data, run the `stream_carla.py` script to begin recording:

```bash
python stream_carla.py \
  --town Town04 \
  --num_frames 600 \
  --out_dir ./data \
  --clip_id 1
```

This will create a 600-frame clip inside `./data/clip_001`. Each recording is saved under:

```plaintext
<out_dir>/clip_<clip_id>/
├── ego_pose/              # 4×4 SE(3) world←ego transforms (NumPy .npy)
├── lidar/                 # LiDAR point clouds with intensity (NumPy .npz)
├── camera_front/          # RGB camera frames (PNG images)
├── labels_3d/             # 3D dynamic object bounding boxes (JSON)
├── hdmap/static_map.xodr  # OpenDRIVE static HD map
├── calibration/           # Sensor calibration files (lidar.json, camera_front.json)
├── timestamp.json         # Frame timestamps in seconds
├── record.log             # Log file of the recording session
└── camera_front.mp4       # (Optional) Rendered video if `--make_video` is used
```

Note that:

* only **three blobs are written per frame** — the rest are static across the clip.
* All coordinates follow the **right-handed ENU** convention (like Waymo & RDS-HQ).
