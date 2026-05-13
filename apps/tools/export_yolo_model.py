"""Export YOLO model to ONNX / TensorRT engine for edge deployment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def export_onnx(model_path: str, output_dir: str, imgsz: int, half: bool) -> Path:
    """Export a YOLO model to ONNX format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install -e '.[perception]'", file=sys.stderr)
        sys.exit(1)

    model = YOLO(model_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {model_path} -> ONNX (imgsz={imgsz}, half={half})")
    export_path = model.export(format="onnx", imgsz=imgsz, half=half)
    exported = Path(export_path)

    # Move to output dir if needed
    dest = out_dir / exported.name
    if exported != dest and exported.exists():
        dest.write_bytes(exported.read_bytes())
        print(f"ONNX model saved to {dest}")
    else:
        print(f"ONNX model saved to {exported}")

    return dest if dest.exists() else exported


def build_engine(onnx_path: str, output_path: str | None = None, fp16: bool = True) -> str:
    """Build TensorRT engine from ONNX using trtexec."""
    try:
        subprocess.run(["trtexec", "--help"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: trtexec not found. Install TensorRT tools.", file=sys.stderr)
        sys.exit(1)

    onnx = Path(onnx_path)
    if not onnx.exists():
        print(f"ERROR: ONNX file not found: {onnx_path}", file=sys.stderr)
        sys.exit(1)

    engine_path = output_path or str(onnx.with_suffix(".engine"))
    cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if fp16:
        cmd.append("--fp16")

    print(f"Building TensorRT engine: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: trtexec failed with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"TensorRT engine saved to {engine_path}")
    return engine_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO model for edge deployment")
    parser.add_argument("--model-path", required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for export")
    parser.add_argument("--half", action="store_true", default=True, help="Export FP16 ONNX")
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--build-engine", action="store_true", help="Also build TensorRT engine via trtexec")
    parser.add_argument("--engine-output", default=None, help="Output path for .engine file")
    args = parser.parse_args()

    onnx_path = export_onnx(args.model_path, args.output_dir, args.imgsz, args.half)

    if args.build_engine:
        build_engine(str(onnx_path), args.engine_output, fp16=args.half)


if __name__ == "__main__":
    main()
