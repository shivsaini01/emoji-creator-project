import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the model
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Load trained weights
emotion_model.load_weights('emotion_model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprised"
}

emoji_dist = {
    0: "./emojis/angry.png", 1: "./emojis/disgusted.png", 2: "./emojis/fearful.png",
    3: "./emojis/happy.png", 4: "./emojis/neutral.png", 5: "./emojis/sad.png",
    6: "./emojis/surpriced.png"
}

last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
cap1 = None
show_text = [3]

# Detect available camera index
def get_available_camera_index():
    for i in range(5):
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            test_cap.release()
            return i
    return -1

def show_vid():
    global cap1, last_frame1

    if cap1 is None:
        cam_index = get_available_camera_index()
        if cam_index == -1:
            print("❌ No working camera found.")
            return
        cap1 = cv2.VideoCapture(cam_index)

    flag1, frame1 = cap1.read()

    if not flag1:
        print("❌ Failed to grab frame")
        return

    frame1 = cv2.resize(frame1, (600, 500))

    bounding_box = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        show_text[0] = maxindex
        cv2.rectangle(frame1, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        cv2.putText(frame1, emotion_dict[maxindex], (x + 20, y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    last_frame1 = frame1.copy()
    pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_vid)

def show_vid2():
    try:
        frame2 = cv2.imread(emoji_dist[show_text[0]])
        img2 = Image.fromarray(frame2)
        imgtk2 = ImageTk.PhotoImage(image=img2)
        lmain2.imgtk2 = imgtk2
        lmain2.configure(image=imgtk2)
        lmain3.configure(text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))
    except Exception as e:
        print("⚠️ Emoji image load error:", e)
    lmain2.after(10, show_vid2)

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root.configure(bg='black')

    try:
        img = ImageTk.PhotoImage(Image.open("logo.png"))
        heading = Label(root, image=img, bg='black')
        heading.pack()
    except:
        heading = Label(root, text="EMOJI CREATOR", pady=20, font=('arial', 45, 'bold'), bg='black', fg='#CDCDCD')
        heading.pack()

    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')

    lmain.place(x=50, y=250)
    lmain2.place(x=900, y=350)
    lmain3.place(x=960, y=250)

    Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold')).pack(side=BOTTOM)

    show_vid()
    show_vid2()
    root.mainloop()
