from model import build_model, checkpoint_path, ys
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model

model = build_model()
model.compile(optimizer='adam', loss="mse")

model.load_weights(filepath=checkpoint_path)
# model.save(filepath='save/model1.h5')

wheel = cv2.imread('steering_wheel_image.jpg')

wheel = cv2.resize(wheel, (270, 210))
rows, cols, level = wheel.shape

i = 0
smooth_angle = 0
while(cv2.waitKey(10) != ord('q')):
    image = cv2.imread('driving_dataset/'+str(i) + ".jpg")
    image_show = image
    image = cv2.resize(image[-150:], (200, 66))
    image = img_to_array(image)/255

    cv2.imshow("Self Driving Car", cv2.resize(image_show, (800, 398)))

    result = -model.predict(image[None])*180.0/3.14159265
    print("Actual Angle= {} Predicted Angle= {}".format(
        str(ys[i]), str(-result)))

    # this section just for the smoother rotation of streeing wheel.
    smooth_angle += 0.2 * pow(abs((result - smooth_angle)),
                              2.0/3.0)*(result - smooth_angle)/abs(result-smooth_angle)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), float(smooth_angle), 1)
    dst = cv2.warpAffine(wheel, M, (cols, rows))

    cv2.imshow("Wheel", dst)
    i += 1


cv2.destroyAllWindows()
