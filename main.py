from multiprocessing import Process, Queue
import cv2
import numpy as np
import base64
import pygame

def startbgmusic(track):
    pygame.mixer.init()
    pygame.mixer.music.load(track)
    pygame.mixer.music.set_volume(0.3)
    # Play the music file indefinitely (the argument -1 means looping forever)
    pygame.mixer.music.play(-1)
    # Keep the program running to play music
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
        
def music_process():
    startbgmusic("bg.mp3")

def enhance_image_contrast_saturation(image):
    # Convert to float to prevent clipping values    
    image = np.float32(image) / 255.0
    
    # Adjust contrast (1.0-3.0)
    contrast = 1.5
    image = cv2.pow(image, contrast)
    
    # Convert to HSV color space to adjust saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Adjust saturation (1.0-3.0)
    saturation_scale = 1.15
    hsv[:, :, 1] *= saturation_scale

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Clip the values and convert back to uint8
    enhanced_image = np.clip(enhanced_image, 0, 1)
    enhanced_image = (255 * enhanced_image).astype(np.uint8)

    return enhanced_image


def webcam_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return
    
    cv2.namedWindow("Sayonara Hokagee", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Sayonara Hokage", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        frame = enhance_image_contrast_saturation(frame)
        cv2.imshow("Sayonara Hokage", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
def resize_image(image, max_width = 500):
    # make sure the aspect ratio is maintained
    h, w = image.shape[:2]
    
    ratio = max_width / float(w)
    new_height = int(h * ratio)
    
    # this will resize the image to have a height of 500px
    resized_image = cv2.resize(
        image, (max_width, new_height), interpolation=cv2.INTER_AREA
    )
    return resized_image

def pass_to_gpt(image, script):
    # pass the image and script to GPT-3
    return "GPT-4 output"
    
    
def process_frames(queue):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Webcam not accessible in process_frames.")
        return

    frames_count = 0
    script = []
    
    while True:
        ret, frame = cap.read() 
        frame = cv2.flip(frame, 1)
        if not ret:
            break
        print("----capturing----")
        
        filename = "frame.jpg"
        
        cv2.imwrite(filename, frame)
        
        resized_frame = resize_image(frame)
        retval, buffer = cv2.imencode(".jpg", resized_frame)
        # convert to lot of text which represents the image
        base64_image = base64.b64encode(buffer).decode("utf-8")
        openai_output = pass_to_gpt(base64_image, script)
        script = script + [{"role": "assistant", "content": openai_output}]
        
        frames_count += 1
        print("script:", script)
        # put the output in the queue for the main process
        queue.put(openai_output)
        #TODO: now play audio here, the script will be our openai_outut
    
    cap.release()


def main():
    queue = Queue()
    webcam_process = Process(target=webcam_capture, args=())
    music_proc = Process(target=music_process)
    frames_process = Process(target=process_frames, args=(queue,))


    webcam_process.start()
    frames_process.start()
    music_proc.start()

    webcam_process.join()
    frames_process.join()
    music_proc.join()

if __name__ == "__main__":
    main()
