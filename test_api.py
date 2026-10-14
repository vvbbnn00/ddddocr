"""HTTP contract and bounded-overload regression; Python standard library only."""
import argparse
import base64
import concurrent.futures
import http.client
import json
import pathlib
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent


def request(base, path, data=None, content_type='application/json', method=None):
    if isinstance(data, dict):
        data = json.dumps(data).encode()
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={'Content-Type': content_type})
    try:
        response = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        raw = response.read()
        return response.status, json.loads(raw) if raw else None


def multipart(files, fields=None):
    boundary = 'ddddocr-test-boundary'
    parts = []
    for name, value in (fields or {}).items():
        parts += [('--' + boundary + '\r\nContent-Disposition: form-data; name="' + name + '"\r\n\r\n' + str(value) + '\r\n').encode()]
    for name, value in files.items():
        parts += [('--' + boundary + '\r\nContent-Disposition: form-data; name="' + name + '"; filename="image.png"\r\nContent-Type: image/png\r\n\r\n').encode(), value, b'\r\n']
    parts += [('--' + boundary + '--\r\n').encode()]
    return b''.join(parts), 'multipart/form-data; boundary=' + boundary


def run(base):
    image = (ROOT / 'image/1.png').read_bytes()
    encoded = base64.b64encode(image).decode()
    code, reference = request(base, '/ocr', {'image': encoded})
    assert code == 200 and isinstance(reference['result'], str), (code, reference)
    assert set(reference) == {'result', 'processing_time'}
    for path in ['/ocr', '/ocr/binary']:
        code, result = request(base, path, image, 'application/octet-stream')
        assert code == 200 and result['result'] == reference['result'], (path, code, result)
    for path in ['/ocr', '/ocr/file']:
        data, mime = multipart({'file': image}, {'probability': 'false', 'colors': '[]'})
        code, result = request(base, path, data, mime)
        assert code == 200 and result['result'] == reference['result'], (path, code, result)
    data, mime = multipart({'file': image, 'image': image})
    assert request(base, '/ocr/file', data, mime)[0] == 400
    filtered = request(base, '/ocr', {'image': encoded, 'color_filter': 'green'})[1]['result']
    data, mime = multipart({'file': image}, {'color_filter': 'green'})
    assert request(base, '/ocr/file', data, mime)[1]['result'] == filtered
    assert request(base, '/ocr', json.dumps({'image': encoded}).encode(), 'application/json-invalid')[0] == 415
    code, result = request(base, '/ocr', {'image': 'data:image/png;base64,' + encoded, 'probability': True})
    assert code == 200 and result['result']['text'] == reference['result'], (code, result)
    probabilities = result['result']['probabilities']
    assert probabilities and isinstance(probabilities[0][0], list)
    assert abs(sum(probabilities[0][0]) - 1) < 1e-5
    assert 0 <= result['result']['confidence'] <= 1
    code, result = request(base, '/ocr?beta=true', {'image': encoded})
    assert code == 200 and isinstance(result['result'], str)
    code, result = request(base, '/ocr', {'image': encoded, 'charset_range': 0})
    assert code == 200 and all(c in '0123456789' for c in result['result'])
    code, _ = request(base, '/set_charset_range', {'charset_range': ['0','1']})
    assert code == 200
    code, result = request(base, '/ocr', {'image': encoded})
    assert code == 200 and all(c in '01' for c in result['result'])
    code, _ = request(base, '/set_charset_range', {'charset_range': []})
    assert code == 200
    assert request(base, '/ocr', {'image': encoded})[1]['result'] == reference['result']
    for path, payload in [('/ocr', {'image': '!'}), ('/ocr', {}), ('/ocr', {'image': 'YQ=='}), ('/ocr?use_gpu=true', {'image': encoded}), ('/ocr?ocr=false', {'image': encoded}), ('/ocr?det=true', {'image': encoded}), ('/ocr', {'image': encoded, 'colors': ['not-a-color']})]:
        code, result = request(base, path, payload)
        assert code == 400 and isinstance(result['detail'], str), (path, code, result)
    assert request(base, '/ocr', b'a', 'text/plain')[0] == 415
    assert request(base, '/ocr/binary', b'x' * (8 * 1024 * 1024 + 1), 'image/png')[0] == 413
    assert request(base, '/health')[0] == 200
    assert request(base, '/config')[1]['default_config']['show_ad'] is False
    assert '/ocr/binary' in request(base, '/docs.json')[1]['paths']
    # Detection must agree across transports.
    detection_image = (ROOT / 'image/5.jpg').read_bytes()
    code, detection = request(base, '/det', {'image': base64.b64encode(detection_image).decode()})
    assert code == 200 and isinstance(detection['result'], list)
    data, mime = multipart({'file': detection_image})
    assert request(base, '/det/file', data, mime)[1]['result'] == detection['result']
    assert request(base, '/det/binary', detection_image, 'image/jpeg')[1]['result'] == detection['result']
    for route, target, background in [('slide_match', 'hk.png', 'bg.png'), ('slide_comparison', 'c.jpg', 'd.jpg')]:
        target = (ROOT / 'image' / target).read_bytes()
        background = (ROOT / 'image' / background).read_bytes()
        code, expected = request(base, '/' + route, {'target_image': base64.b64encode(target).decode(), 'background_image': base64.b64encode(background).decode()})
        assert code == 200, (route, expected)
        data, mime = multipart({'target_image': target, 'background_image': background})
        assert request(base, '/' + route + '/file', data, mime)[1]['result'] == expected['result']
    # A burst is either serviced correctly or explicitly rejected; no server errors.
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        responses = list(executor.map(lambda _: request(base, '/ocr/binary', image, 'image/png'), range(64)))
    assert all(code in (200, 503) for code, _ in responses)
    assert all(value['result'] == reference['result'] for code, value in responses if code == 200)
    # Hold every admission slot with incomplete uploads. Health must still respond.
    parsed = urllib.parse.urlparse(base)
    maximum = request(base, '/config')[1]['limits']['max_concurrency']
    sockets = []
    try:
        for _ in range(maximum):
            conn = socket.create_connection((parsed.hostname, parsed.port or 80), timeout=3)
            conn.sendall(('POST /ocr/binary HTTP/1.1\r\nHost: localhost\r\nContent-Type: image/png\r\nContent-Length: 100\r\n\r\na').encode())
            sockets.append(conn)
        for _ in range(100):
            if request(base, '/status')[1]['active_requests'] == maximum:
                break
            time.sleep(0.01)
        assert request(base, '/status')[1]['active_requests'] == maximum
        assert request(base, '/ocr/binary', image, 'image/png')[0] == 503
        start = time.perf_counter()
        assert request(base, '/health')[0] == 200
        health_ms = (time.perf_counter() - start) * 1000
    finally:
        for conn in sockets:
            conn.close()
    for _ in range(100):
        if request(base, '/status')[1]['active_requests'] == 0:
            break
        time.sleep(0.01)
    assert request(base, '/ocr/binary', image, 'image/png')[0] == 200
    print(json.dumps({'http_regression': 'passed', 'burst_requests': len(responses), 'burst_ok': sum(code == 200 for code, _ in responses), 'burst_503': sum(code == 503 for code, _ in responses), 'health_ms_while_full': round(health_ms, 2), 'image_bytes': len(image), 'base64_json_bytes': len(json.dumps({'image': encoded}).encode())}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    run(parser.parse_args().url.rstrip('/'))
