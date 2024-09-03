from multiprocessing import Process, Queue
import cv2
import numpy as np
import base64
import pygame
import time
import requests
import os

OPENAPI_KEY = os.getenv("OPENAI_API_KEY")
your_name = "shrit"

def startbgmusic(track):
    pygame.mixer.init()
    pygame.mixer.music.load(track)
    pygame.mixer.music.set_volume(0.3)
    pygame.mixer.music.play(-1)
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
        
def play_audio(text):
    print("WRITE THE TEXT TO AUDIO CODE HERE")

def music_process():
    startbgmusic("bg.mp3")

def enhance_image_contrast_saturation(image):
    image = np.float32(image) / 255.0
    contrast = 1.5
    image = cv2.pow(image, contrast)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_scale = 1.15
    hsv[:, :, 1] *= saturation_scale
    enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    enhanced_image = np.clip(enhanced_image, 0, 1)
    enhanced_image = (255 * enhanced_image).astype(np.uint8)
    return enhanced_image

def webcam_capture(queue):
    cap = cv2.VideoCapture(0)
    subtitle_text = "---"

    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return
    
    cv2.namedWindow("Sayonara Hokage", cv2.WINDOW_AUTOSIZE)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        frame = enhance_image_contrast_saturation(frame)
        if not queue.empty():
            subtitle_text = queue.get()
        frame = add_subtitle(frame, subtitle_text)
        cv2.imshow("Sayonara Hokage", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def resize_image(image, max_width=500):
    h, w = image.shape[:2]
    ratio = max_width / float(w)
    new_height = int(h * ratio)
    resized_image = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def generate_new_line(base64_image, name=None):
    if name:
        content = [
            {"type": "text", "text": f"Describe this scene like you're a narrator in a movie. The character's name is {name}."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    else:
        content = [
            {"type": "text", "text": "Describe this scene like you're a narrator in a movie"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    
    return {
        "role": "user",
        "content": content
    }


def pass_to_gpt(base64_image, script, name=None):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAPI_KEY}"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": """
You are the narrator of a hero film. The name of each character is below their face. If not, don't name them. Narrate the characters as if you were narrating the main characters in an epic opening sequence. Be sure to call them by their names.
Make it really awesome, while really making the characters feel epic. Don't repeat yourself. Make it short, max one line 10-20 words. Build on top of the story as you tell it. Don't use the word image. 
As you narrate, pretend there is an epic Hans Zimmer song playing in the background.
Use words that are simple but poetic, a 4th grader should be able to understand it perfectly.
Build a back story for each of the characters as the heroes of a world they're trying to save.
                """.strip(),
            },
        ]
        + script
        + [generate_new_line(base64_image, name)],
        "max_tokens": 300,
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()
        if "choices" in response_data:
            return response_data["choices"][0]["message"]["content"]
        else:
            print("Error:", response_data)
            return None
    except requests.exceptions.RequestException as e:
        print("Request error:", e)
        return None


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
        base64_image = base64.b64encode(buffer).decode("utf-8")
        gpt_4_output = pass_to_gpt(base64_image, script)
        script = script + [{"role": "assistant", "content": gpt_4_output}]
        print("script:", script)

        frames_count += 1
        queue.put(gpt_4_output)
        play_audio(gpt_4_output)
        time.sleep(5)
    
    cap.release()

def add_subtitle(image, text="", max_line_length=40):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    shadow_color = (0, 0, 0)
    line_type = 2
    margin = 10
    line_spacing = 30
    shadow_offset = 2

    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + word) <= max_line_length:
            current_line += word + " "
        else:
            lines.append(current_line)
            current_line = word + " "
    lines.append(current_line)

    text_height_total = line_spacing * len(lines)
    start_y = image.shape[0] - text_height_total - margin

    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_scale, line_type)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = start_y + i * line_spacing

        cv2.putText(image, line, (text_x + shadow_offset, text_y + shadow_offset), font, font_scale, shadow_color, line_type)
        cv2.putText(image, line, (text_x, text_y), font, font_scale, font_color, line_type)

    return image

def main():
    queue = Queue()
    webcam_process = Process(target=webcam_capture, args=(queue,))
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
