import cv2
from court import Court, build_H, court_to_image, image_to_court
from io_video import open_video, frame_count, save_nth_frame
from scripts.choose_corners import click_court_corners


image_corners = [(279.6, 502.8), (1002.4, 511.8), (824.7, 167.8), (458.3, 164.8)]

def main():
    video = open_video("tennis.mp4")
    frame_count = frame_count(video)
    image_corners = click_court_corners(video)
    for i in range(frame_count):
        frame = save_nth_frame(video, i)


if __name__ == "__main__":
    main()