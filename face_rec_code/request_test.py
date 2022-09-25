import requests

# test the server requests
port = "3000"
post_obj = {
            "file_name": "3.mp4",
            # "model": "hog",
            "skip_frames": 30,
            # "resiz_factor": 1,
            # "num_jitters": 1,
            # "tolerance": 0.6,
            }
# method = "run_defaults"
method = "run_custom"
host = f'http://localhost:{port}/{method}'
r = requests.post(host, json=post_obj)
# r = requests.get(host)


print(r.status_code)
print(r.text)
