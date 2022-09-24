import requests

# test the server requests
port = "3000"
method = "run_custom"
post_obj = {
            "file_name": "5.mp4",
            # "model": "hog",
            "skip_frames": 60,
            # "resiz_factor": 1,
            # "num_jitters": 1,
            # "tolerance": 0.6,
            }
host = f'http://localhost:{port}/{method}'
r = requests.post(host, json=post_obj)


print(r.status_code)
print(r.text)
