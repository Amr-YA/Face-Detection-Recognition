# Face Recognition
Face recognition model: detects multiple faces in a video and recognize known persons.
faces pictures, names, video has to follow certain directory structure:
```
|-- face_rec_files: root folder mounted to the docker
|    |-- feed: contain the videos for the analysis
|    |-- unknown: the result of the unidentified faces will be stored here
|    |-- known: contain folders for people to recognize
|    |    |-- person_1: Name of person_1, all person_1 photos are inside the folder
|    |    |-- person_2: Name of person_2, all person_2 photos are inside the folder
|    |    |-- person_x: Name of person_x, all person_x photos are inside the folder
```

## Building the docker
`docker build -f ./Dockerfile -t face_rec .`

## Running docker
run the docker with a mount option pointing to a local folder on the host machine, this has to follow the structure mentioned above

`docker run -it -p 3000:3000 --rm --name face_rec --mount type=bind,source=C:/path/to/mounted_folder,target=/app/face_rec_files face_rec`
```
-it: interactive,
-p: exposed port,
--rm: remove after stop,
--name: name of the container,
--mount: mount external location from host to the container
```
## Request service
3 request are available in the docker  
- `healthcheck` - GET
- `run_defaults` - GET
- `run_custom` - POST

request can be used with url: `http://hostip:3000/method_name`

## GET /healthchech

checks if the docker is running and listning to the correct port, and check if the mounted folder exist

**GET /healthchech response**  
```
json = {
        "status": "current status"
        }
& status_code
```

status_code & status values:  
- 200: ready  
- 402: mount folder not found  

## GET /run_defaults
activates the analysis by default parameters and analyze the first video in the feed folder then return response object and the output files

## POST /run_custom
activates the analysis by custom parameters and custom video file from the feed folder then return response object and the output files

**POST /run_custom request object**  
A json object containing all the required parameters has to be passed through the POST request.  
sample request: `requests.post(host, json=parameters_json)`

parameters_json keys and values:
not all keys has to be passed, the server will replace the missing keys from the request with default values, it's recommended to leave the default values except for the file name and skip frames, the rest is just used to fine tune the matching or more advanced usage.
```
parameters_json = {
            "file_name": "2.mp4", # video file to be used for the analysis from the feed folder
            "skip_frames": 10, # frames to skip while analyzing, higher skip = faster analysis. possible values: [0, 10, 20, ...]
            "resiz_factor": 1, # resize factor for the video image size, lower resiz_factor = faster analysis but lower accurecy. possible values: [0.1, 0.5, 1.5, ...]
            "num_jitters": 1, how many times to sample the known pictures, higher jitter = slower analysis but higher accurecy. possible values: [0, 1, 2, ...]
            "tolerance": 0.6, tolerance for matching faces with known people, higher tolerance = more matches but lower accurecy. possible values: [0, 0.4, 0.8, 1]
            "model": "hog", # model to be used, either "hog" for CPU analysis or "cnn" for GPU, GPU might not be accessible through the docker . possible values: ["hog", "cnn]
            }
```


## /run response
response is in json format accessed from response[text],  
`status`: did the analysis complete successfully  
`video_file`: full directory of file used for the analysis  
`known_faces_count`: number of faces detected and recognized (count duplicates)  
`unknown_faces_count`: number of faces detected but not recognized (count duplicates)  
`total_faces_count`: number of all faces detected (count duplicates)  
`known_faces_list`: list of recognized faces and their first appearance in MSec  
`faces_split_timestamps`: list of recognized faces and every appearance start and end in MSec - faces recognized in a single frame are removed from this list (most likely false detections)  
```
Sample return json = {
        "status":"done",
        "video_file": full/path/video_file.mp4,
        "known_faces_count": 26,
        "unknown_faces_count":18
        "total_faces_count":44,
        "known_faces_list":
            [
                {"name":"thor","time":7465},
                {"name":"loki","time":10385}
            ],
        "faces_split_timestamps": [
        {
            "name": "thor",
            "appearances_count": 2,
            "appearances_details": [
                {
                    "entry_start": 7160,
                    "entry_finish": 7160
                },
                {
                    "entry_start": 14360,
                    "entry_finish": 14360
                }
            ]
        },
        {
            "name": "loki",
            "appearances_count": 1,
            "appearances_details": [
                {
                    "entry_start": 7160,
                    "entry_finish": 7160
                }
            ]
        },
    ]   
    }
& status_code
```

**/run output files**  
files created by the model after it finishes processing:  
- in unknown folder: creates a new folder by the name of the video containing images for each detected but unrecognized face and a labeled video