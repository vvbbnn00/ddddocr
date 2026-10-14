#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path


def ocr_recognition(image_path, endpoint="http://127.0.0.1:8000/ocr", color_filter=None, charset_range=None):
    import requests
    
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    
    parameters = {}
    if color_filter:
        parameters["color_filter"] = color_filter
    if charset_range is not None:
        parameters["charset_range"] = charset_range

    try:
        response = requests.post(
            endpoint, data=image_path.read_bytes(), params=parameters,
            headers={"Content-Type": "application/octet-stream"}, timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        
        return result["result"]
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to connect to ddddocr service: {e}", file=sys.stderr)
        print(f"Make sure the service is running: python scripts/start_ddddocr.py", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="OCR recognition using ddddocr service")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--color-filter", help="Color filter to apply (red, blue, green, yellow, orange, purple, cyan, black, white, gray)")
    parser.add_argument("--charset-range", help="Character range for recognition (0-7 or custom string)")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/ocr", help="OCR endpoint URL")
    parser.add_argument("--text-only", action="store_true", help="Output only the recognized text")
    
    args = parser.parse_args()
    
    result = ocr_recognition(
        args.image_path,
        endpoint=args.endpoint,
        color_filter=args.color_filter,
        charset_range=args.charset_range
    )
    
    if args.text_only:
        print(result)
    else:
        print(f"Text: {result}")


if __name__ == "__main__":
    main()
