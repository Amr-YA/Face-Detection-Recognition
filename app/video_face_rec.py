#%%
import face_recognition
import os
import cv2
import numpy as np
import dlib

MODEL = 'hog'  # 'hog': faster, less acurate - or - 'cnn': slower, more accurate
TOLERANCE = 0.6
APP_DIR = os.path.dirname(__file__)
CURRENT_DIR = os.path.dirname(APP_DIR)
#%%

# confirm required dirs exist and append the current dir to the required folders name to produce absolute paths
def confirm_dirs(parent_dir):
    mount_relative_dir = 'face_rec_files'
    know_faces_relative_dir = 'known'
    unknown_faces_relative_dir = 'unknown'
    video_relative_dir = "feed"

    mount_full_dir = os.path.normpath(os.path.join(parent_dir, mount_relative_dir))
    mount_exist = os.path.isdir(mount_full_dir)

    status_msg = ""

    if not(mount_exist):
        status_msg = "root working directory not found"
        print(status_msg)
        return False, "", "", "", status_msg
    try:
        know_faces_full_dir = os.path.normpath(os.path.join(mount_full_dir, know_faces_relative_dir))
        unknown_faces_full_dir = os.path.normpath(os.path.join(mount_full_dir, unknown_faces_relative_dir))
        video_full_dir = os.path.normpath(os.path.join(mount_full_dir, video_relative_dir))

        know_faces_full_dir_exist = os.path.isdir(know_faces_full_dir)
        unknown_faces_full_dir_exist = os.path.isdir(unknown_faces_full_dir)
        video_full_dir_exist = os.path.isdir(video_full_dir)

        if know_faces_full_dir_exist and unknown_faces_full_dir_exist and video_full_dir_exist:
            status_msg = "Root and child dirs are all correct"
            print(status_msg)
            return True, know_faces_full_dir, unknown_faces_full_dir, video_full_dir, status_msg
        else:
            status_msg = f"known folder: {know_faces_full_dir_exist}, unknown folder: {unknown_faces_full_dir_exist}, video folder: {video_full_dir_exist}"
            print(status_msg)
            return False, "", "", "", status_msg
    except Exception as e:
        print("-----eror: ", e)
        status_msg = f"unexpected errer: {e}"
        print(status_msg)
        return False, "", "", "", status_msg

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

    # TODO
    # if show_video_output: 
    #     cv2.imshow('Video', frame)
    
    return frame

# the inference 
def video_inference(known_encodings, known_names, video_folder, unknown_faces_dir, model, skip_frames, n_upscale, resiz_factor,show_video_output, write_video_output):
    unknowns = 0
    knowns = 0
    faces_in_frames = {}
    frame_array = []
    video_name = os.listdir(video_folder)[0]
    video_file = os.path.normpath(os.path.join(video_folder, video_name)) 
    video_capture = cv2.VideoCapture(video_file)
    frame_counter = 1
    print("-------processing: ", video_file)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not(ret):
            print("Video finished at frame ", frame_counter)
            break

        # Only process every other frame of video to save time
        if frame_counter%skip_frames==0:
            height, width, layers = frame.shape
            size = (width,height)
            timestamp = int(video_capture.get(cv2.CAP_PROP_POS_MSEC))

            time_m_s = f"{timestamp//60000}m:{int((timestamp%60000)/1000)}s"

            fps = video_capture.get(cv2.CAP_PROP_FPS)
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
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                top, right, bottom, left = face_location
                top = int(top//resiz_factor)
                right = int(right//resiz_factor)
                bottom = int(bottom//resiz_factor)
                left = int(left//resiz_factor)
                face_image = frame[top:bottom, left:right]
                name = "Unknown"

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
                    if name not in faces_in_frames.keys():
                        faces_in_frames[name]= timestamp

                else:
                    unknowns+=1   

                    image_name = f"frame_{frame_counter}_unknown_{i}.jpeg"
                    if not(os.path.isdir(unknown_faces_dir)): 
                        os.mkdir(unknown_faces_dir)
                    full_img_dir = os.path.normpath(os.path.join(unknown_faces_dir, image_name))
                    cv2.imwrite(full_img_dir ,face_image)
                face_names.append(name)
                print(f", recognized: {name}", end = '')
            print()
            
            frame = show_labeled_image(frame, show_video_output, n_faces, face_locations, resiz_factor, face_names)

        frame_array.append(frame)
        frame_counter +=1


        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release handle to the webcam

    if write_video_output:
        out_name = f"{video_name.split('.')[0]}_out.avi"
        out_path = os.path.normpath(os.path.join(video_folder, out_name))
        print("----output video: ", out_path)
        out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), int(fps), size)
        for i in range(len(frame_array)):
            # writing to a image array
            out_writer.write(frame_array[i])
        out_writer.release()
    video_capture.release()
    cv2.destroyAllWindows()    

    total = unknowns + knowns
    return faces_in_frames, total, unknowns, knowns

# %%
# activate the inference method and produce the results
def run_analysis(model = 'hog', # 'hog': faster, less acurate - or - 'cnn': slower, more accurate
                skip_frames=3, 
                n_upscale=1, 
                resiz_factor=.25, 
                num_jitters=1,
                show_video_output=False,
                write_video_output=True):

    status, know_faces_dir, unknown_faces_dir, video_dir, msg = confirm_dirs(CURRENT_DIR)
    status_code = 0

    if status:
        video_file = os.listdir(video_dir)[0]     
        video_file = os.path.normpath(os.path.join(video_dir, video_file))  
        try:
            known_encodings, known_names = load_faces(know_faces_dir, model=model, num_jitters=num_jitters)
        except Exception as e:
            status_code = 427 # error in loading faces
            status_msg = "error in loading faces"
            print("-----error: ",e)
            obj = {"status": status_msg,}
            return obj, status_code

        try:    
            faces_in_frames, total, unknowns, knowns = video_inference(
                known_encodings, 
                known_names, 
                video_dir, 
                unknown_faces_dir, 
                model = model, 
                skip_frames=skip_frames, 
                n_upscale=n_upscale, 
                resiz_factor=resiz_factor, 
                show_video_output=show_video_output,
                write_video_output=write_video_output
            )

            faces_list = []
            for key, val in faces_in_frames.items():
                person = {"name": key,
                        "time": val}
                faces_list.append(person)

            status_msg = "done"
            status_code = 200
            obj = {"status": status_msg,
                    "video_file": video_file,
                    "total_faces_count": total,
                    "unknown_faces_count": unknowns,
                    "known_faces_count": knowns,
                    "known_faces_list": faces_list
                    }
            return obj, status_code
        except Exception as e:
            print("-----error: ", e)
            status_code = 428 # error in detecting faces
            status_msg = "error in detecting faces"
            obj = {"status": status_msg,}
            return obj, status_code
    else:
        status_code = 426 # error in loading dirs
        obj = {"status": msg,}
        return obj, status_code


if __name__ == "__main__":
    print("--------------- running main file")

def run():
    obj, status_code = run_analysis(model = 'cnn', # 'hog': faster, less acurate - or - 'cnn': slower, more accurate
                                    skip_frames=3, 
                                    n_upscale=1, 
                                    resiz_factor=1, 
                                    num_jitters=2,
                                    show_video_output=False,
                                    write_video_output=True)

    return obj, status_code

run()  

# %%
