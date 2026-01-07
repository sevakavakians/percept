"""PERCEPT command-line interface."""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PERCEPT - Vision Processing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start PERCEPT")
    start_parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    start_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    start_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    start_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    # Check hardware command
    hw_parser = subparsers.add_parser("check-hardware", help="Check hardware status")

    # Init database command
    db_parser = subparsers.add_parser("init-db", help="Initialize database")
    db_parser.add_argument(
        "--path",
        type=str,
        default="data/percept.db",
        help="Database path",
    )

    # Version command
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "start":
        return cmd_start(args)
    elif args.command == "check-hardware":
        return cmd_check_hardware(args)
    elif args.command == "init-db":
        return cmd_init_db(args)
    elif args.command == "version":
        return cmd_version(args)
    else:
        parser.print_help()
        return 0


def cmd_start(args):
    """Start the PERCEPT server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install uvicorn")
        return 1

    print(f"Starting PERCEPT on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")

    uvicorn.run(
        "ui.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


def cmd_check_hardware(args):
    """Check hardware status."""
    print("=" * 50)
    print("PERCEPT Hardware Check")
    print("=" * 50)

    # Check Hailo
    print("\n[Hailo-8 AI Accelerator]")
    try:
        import subprocess
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("  Status: AVAILABLE")
            for line in result.stdout.strip().split("\n")[:3]:
                print(f"  {line}")
        else:
            print("  Status: NOT AVAILABLE")
            print(f"  Error: {result.stderr.strip()}")
    except FileNotFoundError:
        print("  Status: NOT INSTALLED (hailortcli not found)")
    except Exception as e:
        print(f"  Status: ERROR ({e})")

    # Check RealSense
    print("\n[Intel RealSense Camera]")
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) > 0:
            print(f"  Status: AVAILABLE ({len(devices)} device(s))")
            for i, dev in enumerate(devices):
                print(f"  Device {i}: {dev.get_info(rs.camera_info.name)}")
                print(f"    Serial: {dev.get_info(rs.camera_info.serial_number)}")
        else:
            print("  Status: NO DEVICES FOUND")
    except ImportError:
        print("  Status: NOT INSTALLED (pyrealsense2 not found)")
    except Exception as e:
        print(f"  Status: ERROR ({e})")

    # Check system resources
    print("\n[System Resources]")
    try:
        import psutil
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        print(f"  Memory: {mem.available / (1024**3):.1f}GB available / {mem.total / (1024**3):.1f}GB total")
        print(f"  Disk: {disk.free / (1024**3):.1f}GB free / {disk.total / (1024**3):.1f}GB total")
        print(f"  CPU: {psutil.cpu_count()} cores")
    except ImportError:
        print("  psutil not available")

    # Check temperature (Raspberry Pi)
    print("\n[Temperature]")
    try:
        import subprocess
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"  {result.stdout.strip()}")
        else:
            print("  Not available")
    except FileNotFoundError:
        print("  vcgencmd not found (not a Raspberry Pi?)")
    except Exception:
        print("  Not available")

    print("\n" + "=" * 50)
    return 0


def cmd_init_db(args):
    """Initialize the database."""
    from percept.persistence.database import PerceptDatabase

    db_path = Path(args.path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Initializing database at: {db_path}")

    db = PerceptDatabase(str(db_path))

    # Verify tables exist
    with db._get_connection() as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]

    print(f"Created tables: {', '.join(tables)}")
    print("Database initialized successfully!")
    return 0


def cmd_version(args):
    """Show version."""
    try:
        from importlib.metadata import version
        v = version("percept")
    except Exception:
        v = "0.1.0"

    print(f"PERCEPT v{v}")
    print("Vision Processing Framework for Mobile Robots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
