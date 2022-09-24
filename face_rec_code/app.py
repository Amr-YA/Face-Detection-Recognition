import os
from flask import Flask, request, jsonify
import dlib
from gevent.pywsgi import WSGIServer
from video_face_rec import confirm_dirs, pipeline

import warnings
warnings.simplefilter("ignore")

app = Flask(__name__)
APP_DIR = os.path.dirname(__file__)
DOCKER_DIR = os.path.dirname(APP_DIR)


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    status, _, _, _, _, msg = confirm_dirs(DOCKER_DIR)
    if status:
        status_code = 200
    else:
        status_code = 425 # error in healthcheck
    obj = {"status": msg}
    return jsonify(obj), status_code

@app.route("/run_defaults", methods=["GET"])
def run_defaults():
    print("-----default analysis running")
    obj, status_code = pipeline()
    return jsonify(obj), status_code

@app.route("/run_custom", methods=["POST"])
def run_custom():
    print("-----custom analysis running")

    try:
        file_name = request.json["file_name"]
        print("-----file name found in request: ", file_name)
    except:
        file_name = None
        print("-----file name not found in request")

    try:
        model = request.json["model"]
        print("-----model found in request: ", model)
    except:
        model = 'hog'
        print("-----model name not found in request")

    try:
        skip_frames = request.json["skip_frames"]
        print("-----skip_frames found in request: ", skip_frames)
    except:
        skip_frames = 5
        print("-----skip_frames not found in request")

    try:
        resiz_factor = request.json["resiz_factor"]
        print("-----resiz_factor found in request: ", resiz_factor)
    except:
        resiz_factor = 1
        print("-----resiz_factor not found in request")

    try:
        n_upscale = request.json["n_upscale"]
        print("-----n_upscale found in request: ", n_upscale)
    except:
        n_upscale = 1
        print("-----n_upscale not found in request")

    try:
        num_jitters = request.json["num_jitters"]
        print("-----num_jitters found in request: ", num_jitters)
    except:
        num_jitters = 1
        print("-----num_jitters not found in request")

    try:
        tolerance = request.json["tolerance"]
        print("-----tolerance found in request: ", tolerance)
    except:
        tolerance = 0.6
        print("-----tolerance not found in request")

    obj, status_code = pipeline(video_name=file_name, model=model, skip_frames=skip_frames, resiz_factor=resiz_factor, n_upscale=n_upscale, num_jitters=num_jitters, tolerance=tolerance)
    return jsonify(obj), status_code



if __name__ == "__main__":
    print("-----working on: ",DOCKER_DIR)
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
