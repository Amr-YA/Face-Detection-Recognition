# Face Recognition
Face recognition model: detects multiple faces in a video and recogize known persons.
faces pictures, names, video has to follow certain directory structure:
```
|-- face_rec_files: root folder mounted to the docker
|    |-- feed: contain the video for the analysis
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
`docker run -it -p 3000:3000 --rm --name face_rec --mount type=bind,source=C:/Users/Amr/Desktop/mounted_folder,target=/app/face_rec_files face_rec`
```
-it: interactive,
-p: exposed port,
--rm: remove after stop,
--name: name of the container,
--mount: mount external location from host to the container
```
## Request service
Two request are available in the docker  
- healthcheck  
- run   

both can be used with method `GET` with url: `http://hostip:3000/method_name`

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

## GET /run
activates the analysis and return response object and the output files 

**GET /run response**
```
json = {
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
    } 
& status_code
```

**GET /run output files**  
files created by the model after it finishes processing:  
- in unknown folder: creates an image for each detected but unrecognized face  
- in feed folder: creates a video for faces with labels  
