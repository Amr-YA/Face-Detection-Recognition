# Changelog

## 25-Sep-22 - git:feature-start_end
1. added `video_face_rec.py/faces_split_timestamps` return value for all the enteries a faces appears
2. added `video_face_rec.py/refine_results` to remove single frames faces from enteries return, but not activated in 1st appearance return
3. fixed `timestamp` = 0 by the end of the video
4. fixed `target_fps` having value less than 1 which resulted in corrupted video output

## 24-Sep-22 - git:feature-custm_analysis
1. restructured the files into folders
    - `/face_rec_code/`: for all the python code
    - `/face_rec_files/`: for the videos and pictures
2. split the code into 2 python files
    - `face_rec_code/video_face_rec.py` for loading the folder and files then the analysis code
    - `face_rec_code/app.py` for running the flask server
3. output unknown persons photos and labeled video are saved inside a folder by the name of the video inside `./unknown/`
4. changed `video_face_rec.py/run` to `video_face_rec.py/pipeline` with default parameters in case no parameters passed 
5. added 2 `app.py/run` methods in the server
    - `app.py/run_defaults` - GET: uses the `video_face_rec.py/pipeline` with default parameters, doesn't take any arguments
    - `app.py/run_custom` - POST: uses the `video_face_rec.py/pipeline` with customs parameters, takes the parameters from json object from the post request
6. enhanced the `video_face_rec.py/confirm_dirs` to check all the directories and files are correct
7. added `video_face_rec.py/fetch_video_full_dir` to get the video for the analysis (either from default or custom run)
8. added tolerance parameter in the analysis
9. moved code for saving photos and video to seperate functions `video_face_rec.py/save_photo` & `video_face_rec.py/save_video`
10. changed the flask server running to more stable `app.py/WSGIServer`
11. analysis terminal prints are now in minutes & seconds, ex:"time:  0m:25s - Found 1 faces, recognized: Unknown"