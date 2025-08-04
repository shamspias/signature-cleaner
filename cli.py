import argparse
import sys
from pathlib import Path
import cv2
from typing import Optional
from datetime import datetime

# Import the SignatureProcessor from main.py
# In production, you'd want to refactor this into a separate module
try:
    from main import SignatureProcessor
except ImportError:
    print("Error: Make sure main.py is in the same directory")
    sys.exit(1)


class SignatureCLI:
    """Command line interface for signature processing"""

    def __init__(self):
        self.processor = SignatureProcessor()

    def process_file(
            self,
            input_path: Path,
            output_path: Optional[Path] = None,
            threshold: int = 180,
            smoothing: float = 1.0,
            padding: int = 20,
            invert: bool = False,
            noise_reduction: bool = True,
            format: str = 'png'
    ) -> bool:
        """Process a single signature file"""
        try:
            # Read image
            img = cv2.imread(str(input_path))
            if img is None:
                print(f"❌ Error: Could not read {input_path}")
                return False

            # Process
            processed, metadata = self.processor.process_signature(
                img, threshold, smoothing, padding, invert, noise_reduction
            )

            # Generate output path if not provided
            if output_path is None:
                output_path = input_path.parent / f"{input_path.stem}_cleaned.{format}"

            # Save
            cv2.imwrite(str(output_path), processed)

            print(f"✅ Processed: {input_path.name} -> {output_path.name}")

            if metadata['signature_found']:
                print(f"   Original: {metadata['original_size'][0]}x{metadata['original_size'][1]}")
                print(f"   Processed: {metadata['processed_size'][0]}x{metadata['processed_size'][1]}")
            else:
                print(f"   ⚠️  No signature detected")

            return True

        except Exception as e:
            print(f"❌ Error processing {input_path}: {e}")
            return False

    def process_directory(
            self,
            input_dir: Path,
            output_dir: Optional[Path] = None,
            **kwargs
    ) -> tuple[int, int]:
        """Process all images in a directory"""
        if output_dir is None:
            output_dir = input_dir / "cleaned"

        output_dir.mkdir(exist_ok=True)

        # Find all image files
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(ext))
            image_files.extend(input_dir.glob(ext.upper()))

        if not image_files:
            print(f"No image files found in {input_dir}")
            return 0, 0

        print(f"Found {len(image_files)} images to process")
        print("-" * 50)

        success_count = 0
        for img_path in image_files:
            output_path = output_dir / f"{img_path.stem}_cleaned{img_path.suffix}"
            if self.process_file(img_path, output_path, **kwargs):
                success_count += 1

        print("-" * 50)
        print(f"✅ Successfully processed: {success_count}/{len(image_files)} files")

        return success_count, len(image_files)

    def apply_preset(self, preset_name: str) -> dict:
        """Get preset configuration"""
        presets = {
            'light': {'threshold': 220, 'smoothing': 0.5, 'invert': False},
            'medium': {'threshold': 180, 'smoothing': 1.0, 'invert': False},
            'dark': {'threshold': 140, 'smoothing': 1.5, 'invert': False},
            'pencil': {'threshold': 200, 'smoothing': 0, 'invert': False},
            'scan': {'threshold': 160, 'smoothing': 2.0, 'invert': False}
        }

        if preset_name not in presets:
            print(f"Available presets: {', '.join(presets.keys())}")
            raise ValueError(f"Unknown preset: {preset_name}")

        return presets[preset_name]


def main():
    parser = argparse.ArgumentParser(
        description='Signature Cleaner CLI - Clean digital signatures from command line',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python cli.py signature.jpg

  # Process with custom output
  python cli.py signature.jpg -o cleaned_sig.png

  # Process with custom settings
  python cli.py signature.jpg -t 200 -s 1.5 -p 30

  # Use preset
  python cli.py signature.jpg --preset light

  # Process entire directory
  python cli.py -d ./signatures/ -o ./cleaned/

  # Batch process with preset
  python cli.py -d ./signatures/ --preset pencil
        """
    )

    # Input arguments
    parser.add_argument('input', nargs='?', help='Input image file')
    parser.add_argument('-d', '--directory', help='Process all images in directory')
    parser.add_argument('-o', '--output', help='Output file or directory')

    # Processing options
    parser.add_argument('-t', '--threshold', type=int, default=180,
                        help='Brightness threshold (0-255, default: 180)')
    parser.add_argument('-s', '--smoothing', type=float, default=1.0,
                        help='Smoothing level (0-5, default: 1.0)')
    parser.add_argument('-p', '--padding', type=int, default=20,
                        help='Padding around signature (default: 20)')
    parser.add_argument('--invert', action='store_true',
                        help='Invert colors (for light signatures on dark background)')
    parser.add_argument('--no-noise-reduction', action='store_true',
                        help='Disable noise reduction')

    # Presets
    parser.add_argument('--preset', choices=['light', 'medium', 'dark', 'pencil', 'scan'],
                        help='Use predefined settings')

    # Output options
    parser.add_argument('-f', '--format', choices=['png', 'jpg', 'jpeg'], default='png',
                        help='Output format (default: png)')

    # Other options
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--version', action='version', version='Signature Cleaner CLI v1.0')

    args = parser.parse_args()

    # Validate arguments
    if not args.input and not args.directory:
        parser.error('Either input file or --directory must be specified')

    if args.input and args.directory:
        parser.error('Cannot specify both input file and directory')

    # Create CLI instance
    cli = SignatureCLI()

    # Apply preset if specified
    kwargs = {
        'threshold': args.threshold,
        'smoothing': args.smoothing,
        'padding': args.padding,
        'invert': args.invert,
        'noise_reduction': not args.no_noise_reduction,
        'format': args.format
    }

    if args.preset:
        preset_settings = cli.apply_preset(args.preset)
        kwargs.update(preset_settings)
        if args.verbose:
            print(f"Using preset '{args.preset}': {preset_settings}")

    # Process
    start_time = datetime.now()

    if args.directory:
        # Batch processing
        input_dir = Path(args.directory)
        if not input_dir.exists():
            print(f"Error: Directory {input_dir} does not exist")
            sys.exit(1)

        output_dir = Path(args.output) if args.output else None
        success, total = cli.process_directory(input_dir, output_dir, **kwargs)

        if success < total:
            sys.exit(1)
    else:
        # Single file processing
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: File {input_path} does not exist")
            sys.exit(1)

        output_path = Path(args.output) if args.output else None
        if not cli.process_file(input_path, output_path, **kwargs):
            sys.exit(1)

    # Print timing
    if args.verbose:
        elapsed = datetime.now() - start_time
        print(f"\nTotal time: {elapsed.total_seconds():.2f} seconds")


if __name__ == "__main__":
    main()
