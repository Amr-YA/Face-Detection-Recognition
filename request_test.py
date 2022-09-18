import requests
port = "3000"
method = "run"
host = f'http://localhost:{port}/{method}'
r = requests.get(host)


print(r.status_code)
print(r.text)
