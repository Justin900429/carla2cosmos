# 🚗 Carla2Cosmos

This repository provides a toolkit for converting CARLA simulation data to the RDS-HQ format used by Cosmos. The converted data can be used with [Cosmos Sample AV Transfer](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_sample_av.md) to generate real-world videos with similar traffic scenarios.

>[!CAUTION]
> This repository **DOES NOT** perform direct style transfer from CARLA to real-world video. Instead, it extracts interactive data like bounding boxes and LiDAR points to help generate real-world videos containing similar traffic scenes and behaviors.

| HD Map | Lidar | Generate Output |
| ------ | ----- | --------------- |
|![HD Map](assets/hdmap.gif) | ![lidar](assets/lidar.gif) | ![output](assets/output_video.gif) |

<details>

<summary>Prompt</summary>

The video is captured from a camera mounted on a car. The camera is facing forward. The video showcases a scenic golden-hour drive through a suburban area, bathed in the warm, golden hues of the setting sun. The dashboard camera captures the play of light and shadow as the sun’s rays filter through the trees, casting elongated patterns onto the road. The streetlights remain off, as the golden glow of the late afternoon sun provides ample illumination. The two-lane road appears to shimmer under the soft light, while the concrete barrier on the left side of the road reflects subtle warm tones. The stone wall on the right, adorned with lush greenery, stands out vibrantly under the golden light, with the palm trees swaying gently in the evening breeze. The golden light, combined with the quiet suburban landscape, creates an atmosphere of tranquility and warmth, making for a mesmerizing and soothing drive.
</details>


## ▶️ Recording a Clip

Stream and record synchronous CARLA sessions at **30 Hz**, saving only the essential data needed to convert each clip into the **RDS-HQ** format (used by Cosmos-AV). To record the data, run the `stream_carla.py` script to begin recording:

```bash
python stream_carla.py \
  --town {TOWN_NAME} \
  --num_frames {NUM_FRAMES} \
  --out_dir {DIR_FOR_SAVING_DATA}

# for example
python stream_carla.py \
  --town Town04 \
  --num_frames 600 \
  --out_dir ./data \
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

>[!TIP]
> Run `python stream_carla.py --help` to see all available configuration options, including sensor settings, simulation parameters, and output preferences.

## 🗃️ Converting to RDS-HQ

To convert the recorded data to the RDS-HQ format, run the `convert_to_rds_hq.py` script:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

python toolkit/convert_carla_to_rds_hq.py \
  --root-dir {DIR_WITH_RECORDING_CLIPS} \
  --out-dir {DIR_FOR_SAVING_DATA}

# for example
python toolkit/convert_carla_to_rds_hq.py \
  --root-dir data \
  --out-dir outputs
```

This will create a directory with the following structure:

```plaintext
<out_dir>/
├── pinhole_front/        # Contains video files (.mp4)
├── pinhole_intrinsic/    # Contains camera intrinsic parameters
├── lidar_raw/            # Contains LiDAR data archives (.tar)
├── all_object_info/      # Contains object information data
├── timestamp/            # Contains timestamp data
├── vehicle_pose/         # Contains vehicle pose data
├── pose/                 # Contains pose data
├── 3d_road_boundaries/   # Contains 3D road boundary data
├── 3d_lanelines/         # Contains 3D lane line data
└── 3d_lanes/             # Contains 3D lane data
```

To visualize the converted data:

```bash
python toolkit/visualize_rds_hq.py \
  -i {DIR_FOR_SAVING_DATA} \
  -c {CLIP_ID} \
  -d carla

# for example
python toolkit/visualize_rds_hq.py \
  -i outputs \
  -c 0000 \
  -d carla
```

## 🎥 Rendering from RDS-HQ

>[!TIP]
> The rendering functionality is based on the [cosmos-av-sample-toolkits](https://github.com/nv-tlabs/cosmos-av-sample-toolkits) repository. You can refer to their documentation for more details about the rendering process and options.

To render the HD-Map and Lidar from the RDS-HQ format, run the `render_rds_hq.py` script:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

python toolkit/render_rds_hq.py \
  -d carla \
  -i {DIR_WITH_RDS_HQ_DATA} \
  -o {DIR_FOR_SAVING_RENDERED_DATA} \
  -c ftheta  # recommend

# for example
python toolkit/render_rds_hq.py \
  -d carla -i outputs \
  -o demo_render -c ftheta # recommend
```

This will create a directory with the following structure:

```plaintext
<render_dir>/
demo_render/
├── lidar/
│   └── pinhole_front/    # Contains LiDAR visualization data
└── hdmap/
    └── pinhole_front/    # Contains HD map visualization data
```

For multi-camera setups, rendered data is organized in the `lidar` and
`hdmap` directories with the naming convention `pinhole_<camera_name>`.

>[!Note]
> The current version only supports rendering from the front camera.

## 📝 Development

Please refer to the [Development](DEVELOPMENT.md) for more details.

## 🔒 Immunity

All outputs generated by this system, including but not limited to:

* Recorded data
* Rendered visualizations
* Converted formats
* Derived analytics

are provided for research and educational purposes only. Users are responsible for:

1. Verifying the accuracy and suitability of generated outputs for their specific use case
2. Ensuring compliance with local laws and regulations when using the outputs
3. Obtaining necessary permissions for any commercial or public use
4. Acknowledging that the outputs may not be suitable for safety-critical applications

By using this software, you agree to indemnify and hold harmless the developers and contributors from any claims, damages, or liabilities arising from the use of the models or generated outputs.
