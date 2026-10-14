"""Reproducible HTTP throughput measurement; reports overload separately."""
import argparse
import concurrent.futures
import json
import pathlib
import statistics
import time
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from test_api import request


def run(args):
    image = pathlib.Path(args.image).read_bytes()
    for _ in range(5):
        assert request(args.url, '/ocr/binary', image, 'image/png')[0] == 200
    def one(_):
        start = time.perf_counter()
        code, _ = request(args.url, '/ocr/binary', image, 'image/png')
        return code, (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(one, range(args.requests)))
    elapsed = time.perf_counter() - start
    success = sorted(ms for code, ms in results if code == 200)
    print(json.dumps({'requests': args.requests, 'client_concurrency': args.concurrency, 'success': len(success), 'rejected_503': sum(code == 503 for code, _ in results), 'other_errors': sum(code not in (200,503) for code, _ in results), 'elapsed_seconds': round(elapsed,3), 'successful_requests_per_second': round(len(success)/elapsed,2), 'p50_ms': round(statistics.median(success),2) if success else None, 'p95_ms': round(success[min(len(success)-1,int(len(success)*.95))],2) if success else None}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--image', default=str(pathlib.Path(__file__).resolve().parents[1] / 'image/1.png'))
    parser.add_argument('--requests', type=int, default=200)
    parser.add_argument('--concurrency', type=int, default=4)
    run(parser.parse_args())
