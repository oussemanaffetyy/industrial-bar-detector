import argparse

parser = argparse.ArgumentParser(description='Bar detector application')
parser.add_argument('--video', help='Path to video')
parser.add_argument('--camera', action='store_true', help='use live camera')
args = parser.parse_args()

if args.camera:
    print("Using camera...")
if args.video:
    print(f"Using video: {args.video}")