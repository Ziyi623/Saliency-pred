import cv2

if __name__ == "__main__":
    img = cv2.imread("../data/TrainSet/Stimuli/Action/001.jpg")
    print((img.shape[0],img.shape[1]))
    if img is None:
        print("No")
    else:
        print("Yes")


