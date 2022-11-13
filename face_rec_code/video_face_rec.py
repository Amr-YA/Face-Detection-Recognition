import face_recognition
import os
import cv2
import numpy as np
import dlib
# import time

APP_DIR = os.path.dirname(__file__)
DOCKER_DIR = os.path.dirname(APP_DIR)

# confirm required dirs exist and append the current dir to the required folders name to produce absolute paths
def confirm_dirs(parent_dir, video_name):
    mount_relative_dir = 'face_rec_files'
    know_faces_relative_dir = 'known'
    unknown_faces_relative_dir = 'unknown'
    video_relative_dir = "feed"

    mount_full_dir = os.path.normpath(os.path.join(parent_dir, mount_relative_dir))
    mount_exist = os.path.isdir(mount_full_dir)

    status_msg = ""

    #if error in mount folder, exit confirm_dirs and return status False
    if not(mount_exist):
        status_msg = "root working directory not found"
        print(status_msg)
        return False, "", "", "", "", status_msg

    # load all directories and get full paths
    try:
        know_faces_full_dir = os.path.normpath(os.path.join(mount_full_dir, know_faces_relative_dir))
        unknown_faces_full_dir = os.path.normpath(os.path.join(mount_full_dir, unknown_faces_relative_dir))
        video_folder_full_dir = os.path.normpath(os.path.join(mount_full_dir, video_relative_dir))

        know_faces_full_dir_exist = os.path.isdir(know_faces_full_dir)
        unknown_faces_full_dir_exist = os.path.isdir(unknown_faces_full_dir)
        video_folder_full_dir_exist = os.path.isdir(video_folder_full_dir)
        all_dir_exist = know_faces_full_dir_exist and unknown_faces_full_dir_exist and video_folder_full_dir_exist
    except Exception as e:
        print("-----error11: ", e)
        status_msg = "unexpected error, please check all directories and videos"
        return False, "", "", "", "", status_msg

    #if error in directories structure, exit confirm_dirs and return status False
    if not(all_dir_exist):
        status_msg = f"Error in dirs - known: {know_faces_full_dir_exist}, unknown: {unknown_faces_full_dir_exist}, video: {video_folder_full_dir_exist}"
        print(status_msg)
        return False, "", "", "", "", status_msg       

    status, video_name = fetch_video_full_dir(video_folder_full_dir, video_name)

    #if no video found, exit confirm_dirs and return status False
    if not(status):
        status_msg = "video file not found"
        print(status_msg)
        return False, "", "", "", "", status_msg

    # finally if no error found, return all full paths
    status_msg = "Root and child dirs are all correct"
    print(status_msg)
    return True, know_faces_full_dir, unknown_faces_full_dir, video_folder_full_dir, video_name, status_msg

# get video full path by name or get 1st video in the folder
def fetch_video_full_dir(video_dir, video_name):
    video_dir_files = os.listdir(video_dir)
    print("-----video for analysis", video_dir_files[0] if video_name==None else video_name)
    if len(video_dir_files) > 0:
        video_name = video_dir_files[0] if video_name==None else video_name
        video_file_full_dir = os.path.normpath(os.path.join(video_dir, video_name)) 
        status = os.path.isfile(video_file_full_dir)
    else:
        video_file_full_dir = "empty"
        status = False
    return status, video_name

# load the faces from the known_face_dir folder and get their names
def load_faces(known_face_dir, model, num_jitters):
    known_encodings = []
    known_names = []
    for name in os.listdir(known_face_dir):
        current_person_dir = os.path.normpath(os.path.join(known_face_dir, name))
        # Load every file of faces of known person
        for filename in os.listdir(current_person_dir):
            current_person_face_dir = os.path.normpath(os.path.join(current_person_dir, filename))
            # Load an image
            image = face_recognition.load_image_file(current_person_face_dir)

            # Encoding, return list encoding for EACH face found in photo
            encoding = face_recognition.face_encodings(image, num_jitters=num_jitters, model=model)[0]

            # Append encodings and name
            known_encodings.append(encoding)
            known_names.append(name)
    return known_encodings, known_names

# used to label the frames for faces and show the current frame on screen
def show_labeled_image(frame, show_video_output, n_faces, face_locations, resiz_factor, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        top = int(top//resiz_factor)
        right = int(right//resiz_factor)
        bottom = int(bottom//resiz_factor)
        left = int(left//resiz_factor)
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, f"{n_faces} Faces", (12, 40), font, 0.7, (255, 255, 255), 1)

    # Display the resulting image

    if show_video_output: 
        cv2.imshow('Video', frame)
    
    return frame

# the inference 
def video_inference(known_encodings, known_names, video_folder, video_name, unknown_faces_dir, model, skip_frames, n_upscale, resiz_factor,show_video_output, write_video_output, tolerance):
    unknowns = 0
    knowns = 0
    video_file = os.path.normpath(os.path.join(video_folder, video_name))
    video_name = video_name.split('.')[0]
    faces_in_frames = {}
    faces_all_timestamps = {}
    frame_array = []
    video_capture = cv2.VideoCapture(video_file)
    totalNoFrames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    

    frame_counter = 1
    print("-------processing: ", video_file)

    while True:
        # Grab a single frame of video
        frame_exist, frame = video_capture.read()
        timestamp = int(video_capture.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp == 0:
            timestamp = int(1000 * (totalNoFrames / fps))
        time_m_s = f"{timestamp}MSec : {timestamp//60000}m:{int((timestamp%60000)/1000)}s"
        if not(frame_exist):
            print("Video finished at ", time_m_s)
            break

        # Only process every other frame of video to save time
        if frame_counter%skip_frames==0:
            height, width, layers = frame.shape
            size = (width,height)

            
            print(f"time:  {time_m_s}", end ='')
            # Resize frame of video to 1/4 size for faster face recognition processing
            resized_frame = cv2.resize(frame, (0, 0), fx=resiz_factor, fy=resiz_factor)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            # RGB_frame = resized_frame[:, :, ::-1]
            RGB_frame = resized_frame
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(RGB_frame, number_of_times_to_upsample=n_upscale, model=model)
            face_encodings = face_recognition.face_encodings(RGB_frame, face_locations)
            n_faces = len(face_locations)
            print(f" - Found {n_faces} faces", end ='')

            face_names = []
            for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
                top, right, bottom, left = face_location
                top = int(top//resiz_factor)
                right = int(right//resiz_factor)
                bottom = int(bottom//resiz_factor)
                left = int(left//resiz_factor)
                face_image = frame[top:bottom, left:right]

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    knowns+=1
                    # create two dic, faces_in_frames: first time face appear in video, faces_all_timestamps: all times face appear in video
                    if name not in faces_in_frames.keys():
                        faces_all_timestamps[name] = [timestamp]
                        faces_in_frames[name] = timestamp
                    else:
                        faces_all_timestamps[name].append(timestamp)

                else:
                    name = "Unknown"
                    unknowns+=1   
                    image_name = f"time{timestamp}_unknown.jpeg"
                    save_photo(unknown_faces_dir, video_name, face_image, image_name)
                face_names.append(name)
                print(f", recognized: {name}", end = '')
            print()
            
            frame = show_labeled_image(frame, show_video_output, n_faces, face_locations, resiz_factor, face_names)
            frame_array.append(frame)

        frame_counter +=1


        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    faces_split_timestamps = split_show_time(faces_all_timestamps)

    # fps for output file, if it's less than 1 the files is corrupted
    if skip_frames > 0.5*fps:
        target_fps=int(2*fps/skip_frames)
    else:
        target_fps=int(fps/skip_frames)
    if target_fps < 2: # min fps = 2
        target_fps = 2
    # writes file
    if write_video_output:
        save_video(video_name, unknown_faces_dir, frame_array, target_fps=target_fps, image_size=size)

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()    

    total = unknowns + knowns
    return faces_in_frames, total, unknowns, knowns, faces_split_timestamps

# save photos in folder
def save_photo(unknown_faces_dir, video_name, face_image, image_name):
    unkowndir_per_video_dir = os.path.normpath(os.path.join(unknown_faces_dir, video_name))
    if not(os.path.isdir(unkowndir_per_video_dir)): 
        os.mkdir(unkowndir_per_video_dir)
    full_img_dir = os.path.normpath(os.path.join(unkowndir_per_video_dir, image_name))
    cv2.imwrite(full_img_dir ,face_image)

# save video in folder
def save_video(video_name, video_folder, frame_array, target_fps, image_size):
    out_path = os.path.normpath(os.path.join(video_folder, video_name))
    if not(os.path.isdir(out_path)): 
        os.mkdir(out_path)
    out_name = f"{video_name}_labeled.avi"
    out_path = os.path.normpath(os.path.join(out_path, out_name))
    print("----output video: ", out_path, "fps: ", {target_fps})
    out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), target_fps, image_size)
    for i in range(len(frame_array)):
        # writing to a image array
        out_writer.write(frame_array[i])
    out_writer.release()

# split continous milliseconds into seperate entries of start and end
def split_show_time(named_show, exit_threshold_ms = 1500):
    named_split_shows = {}
    for name, arr in named_show.items():
        temp = [arr[0]]
        splits = []

        for i, v in enumerate(arr):
            if v == temp[-1]:  
                pass
            elif (v - temp[-1] <= exit_threshold_ms):
                temp.append(v)
            elif not(i == len(arr)):
                splits.append(temp)
                temp = [v]
            else:
                splits.append(temp)
        
        # append last temp after for exit
        splits.append(temp)

        # keep start and end only, append dummy value if an array has single frame
        for i, times in enumerate(splits):
            if len(times) == 1:
                times.append(times[-1])
            splits[i] = [times[0], times[-1]]
            
        named_split_shows[name] = splits
    
    named_split_shows = refine_results(named_split_shows)
    return named_split_shows

# remove single frame faces before the split_show_time return results
def refine_results(named_split_shows):
    for name, arrs in list(named_split_shows.items()):
        if (len(arrs) == 1) and (arrs[0][0] == arrs[0][-1]):
            named_split_shows.pop(name)

    return named_split_shows

# activate the inference method and produce the results
def pipeline(
            model = 'hog', # 'hog': faster, less acurate - or - 'cnn': slower, more accurate
            skip_frames=5, # higher = faster, less acurate
            n_upscale=1, # higher = slower, more accurate (min 1)
            resiz_factor=1, # higher = slower, more accurate (min 0)
            num_jitters=1, # higher = slower, more accurate (min 0)
            tolerance=0.6, # lower = more accurate but less matching (min 0)
            show_video_output=False,
            write_video_output=True,
            video_name = None, 
            ):

    status, know_faces_dir, unknown_faces_dir, video_dir, video_name, msg = confirm_dirs(DOCKER_DIR, video_name)
    status_code = 0

    if status: 
        try:
            known_encodings, known_names = load_faces(know_faces_dir, model=model, num_jitters=num_jitters)
        except Exception as e:
            status_code = 427 # error in loading faces
            status_msg = "error in loading faces"
            print("-----error22: ",e)
            obj = {"status": status_msg,}
            return obj, status_code

        try:  
            faces_in_frames, total, unknowns, knowns ,faces_split_timestamps= video_inference(
                known_encodings=known_encodings, 
                known_names=known_names, 
                video_folder=video_dir, 
                video_name=video_name,
                unknown_faces_dir=unknown_faces_dir, 
                model = model, 
                skip_frames=skip_frames, 
                n_upscale=n_upscale, 
                resiz_factor=resiz_factor, 
                show_video_output=show_video_output,
                write_video_output=write_video_output,
                tolerance=tolerance,
            )

            faces_list = []
            for key, val in faces_in_frames.items():
                person = {"name": key,
                        "time": val}
                faces_list.append(person)

            named_split_shows = []
            for key, value in faces_split_timestamps.items():
                entry = {
                        "name": key ,
                        "appearances_count": len(value),
                        "appearances_details": [],
                        }

                for arr in value:
                    temp = {"entry_start": arr[0], "entry_finish": arr[-1]}
                    entry["appearances_details"].append(temp)

                named_split_shows.append(entry)

            status_msg = "done"
            status_code = 200
            obj = {"status": status_msg,
                    "video_file": video_name,
                    "total_faces_count": total,
                    "unknown_faces_count": unknowns,
                    "known_faces_count": knowns,
                    "known_faces_list": faces_list,
                    "faces_split_timestamps": named_split_shows,
                    }
            return obj, status_code
        except Exception as e:
            print("-----error33: ", e)
            status_code = 428 # error in detecting faces
            status_msg = "error in detecting faces"
            obj = {"status": status_msg,}
            return obj, status_code
    else:
        status_code = 426 # error in loading dirs
        obj = {"status": msg,}
        return obj, status_code


# pipeline(video_name="3.mp4", skip_frames=30, resiz_factor=1, model="hog")  
# pipeline()
