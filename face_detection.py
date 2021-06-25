import cv2

# Load some pre-trained data on face tutorials from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml' )

# Choose an image to detect faces in
# img = cv2.imread( 'headshot.png' )

# To capture video from webcam
webcam = cv2.VideoCapture( 0 )

# Iterate over all frames while webcam is live
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Convert to grayscale
    grayscaled_img = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale( grayscaled_img )

    # Draw rectangles around the faces
    for ( x, y, w, h ) in face_coordinates:
        cv2.rectangle( frame, ( x, y ), ( x + w, y + h ), ( 0, 0, 255 ), 10 )

    # Run app
    cv2.imshow( 'Face Detector', frame )
    key = cv2.waitKey(1)

    # Quit app if Q is pressed
    if key == 81 or key == 113:
        break

webcam.release()