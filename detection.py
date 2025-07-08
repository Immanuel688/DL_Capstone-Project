import torch
import cv2
import numpy as np
from pathlib import Path
import time
import pyttsx3 # library for text to speech
import threading

# =============== CONFIGURATION =================
MODEL_PATH = 'runs/train/yolo_glasses_synthetic/weights/best.pt'
INPUT_TYPE = 'webcam'  # 'image', 'video', or 'webcam'
#SOURCE_PATH = 'data/images/sample.jpg'  # Used if INPUT_TYPE is image or video
OUTPUT_DIR = 'runs/custom_detect'
CONFIDENCE_THRESHOLD = 0.5
SPEAK_INTERVAL = 3  # seconds

# =============== TEXT-TO-SPEECH SETUP ==========
tts_engine = pyttsx3.init()
last_spoken = {"message": None, "time": 0} # inital setting for audio

def speak_message_async(message):
    now = time.time() # Gets the current time in seconfds and also check how long ago lastmessage was spoken
    if message != last_spoken["message"] or (now - last_spoken["time"]) > SPEAK_INTERVAL:
        def speak():
            tts_engine.say(message)
            tts_engine.runAndWait()
        
        threading.Thread(target=speak, daemon=True).start() 
        # this opens a seperate thread for audio to run and does not intefere with video frames and ens once orignial thread ends
        last_spoken["message"] = message
        last_spoken["time"] = now

# =============== FILE NAME UTILITY =============
def get_unique_filename(base_name, ext, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        suffix = f"_{i}" if i > 0 else ""
        candidate = output_dir / f"{base_name}{suffix}.{ext}"
        if not candidate.exists():
            return str(candidate)
        i += 1

# =============== MODEL LOAD ====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
model.to(device).eval()
model.conf = CONFIDENCE_THRESHOLD
model.iou = 0.45 
violation_classes = [0, 1, 2]

# =============== DRAWING FUNCTION ==============
def draw_message(img, x1, y1, x2, y2, violation):
    color = (0, 0, 255) if violation else (0, 255, 0)
    message = "Safety Glass Not Detected" if violation else "Safety Glass Detected"
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
    cv2.putText(img, message, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2, cv2.LINE_AA)

# =============== PROCESS FRAME =================
#def process_frame(frame):
    #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #results = model(rgb)
    #detections = results.xyxy[0].cpu().numpy()
    #messages_in_frame = set()

    #for *box, conf, cls in detections:
        #x1, y1, x2, y2 = map(int, box)
        #cls = int(cls)
        #violation = cls in violation_classes
        #draw_message(frame, x1, y1, x2, y2, violation)
        #msg = "Safety glass not detected" if violation else "Safety glass detected"
        #messages_in_frame.add(msg)

    #for msg in messages_in_frame:
        #speak_message_async(msg)

    #return frame
def process_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb)
    detections = results.xyxy[0].cpu().numpy() # output of detection has bbox coordinates class id and confidence score
    # xyxy[0] is kind of getting the first index value in the list (mostly we deal as batch=1 which takes one image of 640*480 later yolo converts to 640*640
    # conversion of gpu tensor arrays to cpu numpy arrays works well with open CV

    has_violation = False
    has_compliant = False # these are initial settings 

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        cls = int(cls)
        violation = cls in violation_classes  # Violating object
        draw_message(frame, x1, y1, x2, y2, violation)

        if violation:
            has_violation = True
        else:
            has_compliant = True

    # Message logic
    if has_violation and not has_compliant:
        speak_message_async("Safety glass not detected")
    elif has_violation and has_compliant:
        speak_message_async("Warning: Unsafe glasses present along with safety glasses")
    elif has_compliant:
        speak_message_async("Safety glass detected")
    else:
        pass  # No detections

    return frame

# =============== PROCESS IMAGE =================
if INPUT_TYPE == 'image':
    img = cv2.imread(SOURCE_PATH)
    result_img = process_frame(img.copy())
    out_path = get_unique_filename("output", "jpg", OUTPUT_DIR)
    cv2.imwrite(out_path, result_img)
    print(f"[✓] Image saved at {out_path}")
    cv2.imshow("Image Detection", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =============== PROCESS VIDEO/WEBCAM ==========
elif INPUT_TYPE in ['video', 'webcam']:
    cap = cv2.VideoCapture(0 if INPUT_TYPE == 'webcam' else SOURCE_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_path = get_unique_filename("output", "mp4", OUTPUT_DIR)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Four Character Code
    #*'mp4v' unpacks the string 'mp4v' into four separate characters:
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f" Saving to {out_path} — Press 'q' to stop.")

    while cap.isOpened(): # loop should be unlimited
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_frame(frame)
        out_writer.write(processed)
        cv2.imshow("Video Detection", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release() # releasing and closing all windows
    out_writer.release()
    cv2.destroyAllWindows()
    try:
        tts_engine.stop()
    except:
        pass
    print(f" Video saved at {out_path}")

else:
    print(" Invalid INPUT_TYPE. Use 'image', 'video', or 'webcam'.")
