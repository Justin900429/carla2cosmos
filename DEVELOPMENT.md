# 🚀 Development

## 📁 File Structure

```plaintext
carla2cosmos/
├── config/               # Configuration files
├── manager/              # Project management modules
├── misc/                 # Miscellaneous utilities
├── opendriveparser/      # OpenDRIVE parser implementation
├── toolkit/              # Tool utilities
├── utils/                # Utility functions
├── stream_carla.py       # Streaming implementation
├── .gitignore            # Git ignore rules
├── .python-version       # Python version specification
├── DEVELOPMENT.md        # Development documentation
├── README.md             # Project documentation
├── pyproject.toml        # Project dependencies and metadata
└── uv.lock               # Dependency lock file
```

## 📚 Code Attribution

### Adapted from [cosmos-av-sample-toolkits](https://github.com/nv-tlabs/cosmos-av-sample-toolkits)

* `utils/*` (except `utils/opendrive_parser.py`)
* `toolkit/*` (except `toolkit/convert_carla_to_rds_hq.py`)
* `config/*` (except `config/stream_config.py`)

### Adapted from [parse-and-visualize-xodr](https://github.com/skx6/parse-and-visualize-xodr/tree/main)

* `opendriveparser/*`
* `utils/opendrive_parser.py`

## 📝 Notes

### Data Format

The following data are stored in the **global coordinate system** (world frame):

* Ego vehicle poses
* Object bounding boxes
* Camera poses and calibration
* HD Map elements (lanes, boundaries, etc.)

The following data are stored in their **local sensor coordinate system**:

* LiDAR point clouds (in the LiDAR sensor frame)

>[!Caution]
> The original coordinate system is in **left-handed** and we should convert it to the **right-handed** coordinate system (see the following section for details).

### Coordinate System

Since CARLA uses a left-handed coordinate system (inherited from Unreal Engine) but RDS-HQ requires right-handed coordinates, we need to perform coordinate transformations:

For 3D points:

* Simply mirror the Y coordinate by multiplying it by -1
* X and Z coordinates remain unchanged

For rotation matrices:

* Apply a change-of-basis transformation using the mirroring matrix $M_{RH}$:

$$
R_{RH} = M_{RH} R_{LH} M_{RH}^{-1},
$$

where:

* $R_{RH}$ is the rotation matrix in right-handed coordinates
* $R_{LH}$ is the original rotation matrix in left-handed coordinates  
* $M_{RH}$ is the change-of-basis matrix that flips the Y axis

Since $M_{RH}$ is a diagonal matrix with entries [1, -1, 1], it is its own inverse. This simplifies the transformation to:

$$
R_{RH} = M_{RH} R_{LH} M_{RH}
$$
