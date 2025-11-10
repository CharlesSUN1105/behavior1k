from pathlib import Path

models_dir = Path(__file__).parent / "models"
r1pro_dir = models_dir / "r1pro"


r1pro_urdf_path = r1pro_dir / "r1pro_source_color.urdf"  # use for now
# r1pro_urdf_path = r1pro_dir / "r1pro_source.urdf" # when loaded into pybullet is black
