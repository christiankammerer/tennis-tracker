import cv2

def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap  

def frame_count(cap):
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def save_nth_frame(cap, n, output_path):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    return ret

def __main__():
    video = open_video("tennis.mp4")
    frame_number = frame_count(video)
    for i in range(frame_number):
        save_nth_frame(video, i, f"debug/frame_{i}.jpg")

if __name__ == "__main__":
    __main__()