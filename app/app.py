import os
from flask import Flask, request, jsonify
import dlib
from gevent.pywsgi import WSGIServer
from video_face_rec import confirm_dirs, run_quick, run_custom

app = Flask(__name__)
APP_DIR = os.path.dirname(__file__)
CURRENT_DIR = os.path.dirname(APP_DIR)


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    status, _, _, _, msg = confirm_dirs(CURRENT_DIR)
    if status:
        status_code = 200
    else:
        status_code = 425 # error in healthcheck
    obj = {"status": msg}
    return jsonify(obj), status_code


@app.route("/run_quick", methods=["GET"])
def run_quick():

    file_name="lol",
    model = 'hog',
    skip_frames=3, 
    resiz_factor=1,
    num_jitters=1,  
    tolerance=0.6,   


    obj, status_code = run_quick(file_name, model, skip_frames, resiz_factor, num_jitters, tolerance)
    return jsonify(obj), status_code

@app.route("/run_custom", methods=["POST"])
def run_custom():
    obj, status_code = run_custom()
    return jsonify(obj), status_code


if __name__ == "__main__":
    print("-----working on: ",CURRENT_DIR)
    try:
        cuda = dlib.DLIB_USE_CUDA
        if cuda:
            print("-----running on GPU")
        else:
            print("-----running on CPU")
    except:
        print("-----couldnn't detect CUDA")
    
    # app.run(debug=True, host="0.0.0.0", port="3000")

    http_server = WSGIServer(('', 3000), app)
    http_server.serve_forever()
