1. restructured the files into folders
    - /app/: for all the python code
    - /face_rec_files/: for the videos and pictures
2. split the code into 2 python files
    - app/video_face_rec.py for the analysis and loading
    - app/app.py for running the flask server
3. output unknown persons photos and labeled video are saved inside a folder by the name of the video inside /unknown/
4. changed normal `run` to `run_quick` to analyze with default parameters
5. added `run_custom` to analyze with custom parameters
6. enhanced the `confirm_dirs` to check all the directories are correct
7. added `fetch_video_full_dir` to get the video for the analysis (either from default or custom run)
8. added tolerance parameter in the analysis
9. moved code for saving photos and video to seperate functions `save_photo` & `save_video`