import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import tempfile
import time

number_of_demo_image = 6
demo_images = ["images/demo"+str(i)+".jpg" for i in range(1, number_of_demo_image+1)]
demo_videos = ["videos/"+ch for ch in ["africanGirl.mp4","blue.mp4", "free.mp4", "friends.mp4"]]
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Print function costumirize
def html_markdown(label, color='red', text_style = 'center', hmodel = 2):
    st.markdown(f"<h{str(hmodel)} style = 'text-align:{text_style}; color:{color}'>{label}<h{hmodel}/>", unsafe_allow_html=True)


html_markdown("Computer Vision With OpenCV and Mediapipe", hmodel=1, color='black')


st.sidebar.title("SideBar")
st.sidebar.markdown("---")
modes = ["Image", "Video"]
mode = st.sidebar.selectbox("Mode", options=modes)


@st.cache
def img_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    dim = None
    if width is None or height is None:
        if width is None and height is not None:
            dim = (height, w)
        if height is None and width is not None:
            dim = (h, width)
        if width is None and height is None:
            return image
    else:
        dim = (height, width)
    return cv2.resize(image,(dim[1], dim[0]), interpolation=inter)

#def convert_to_hsv():

# Load the image file
@st.cache
def load_file(file, channel):
    val = True
    # Image de l'utilisateur
    if file is not None:
        image = np.array(Image.open(file))
        val = False
        channel = 'RGB'
    # Default Image
    else:
        channel = 'BGR'
        file = demo_image
        image = cv2.imread(file,cv2.IMREAD_COLOR)
    return image, channel

# Convert image to gray
@st.cache
def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert image to hsv
@st.cache
def to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create a mask
@st.cache
def mask(imghsv, hue_min, sat_min, val_min, hue_max, sat_max, val_max):
    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])
    return cv2.inRange(imghsv, lower, upper)

@st.cache
def color_mask(image, mask):
    return cv2.bitwise_and(image, image, mask= mask)
# Send the interface into 2 columns
left_column, right_column = st.columns(2)

# Blur Image Function
@st.cache
def blur_image(image, x):
    return cv2.blur(image, (x,x))

# Median Blur Image
@st.cache
def median_blur_image(image, x):
    return cv2.medianBlur(image, x)

# Gaussian blur image
@st.cache
def gaussian_blur_image(image, x, sigmaX, sigmaY):
    return cv2.GaussianBlur(image, (x,x), sigmaX, sigmaY)

# Erode image function
@st.cache
def img_erosion(image, iterations):
    kernel = np.ones((5, 5), np.uint8)
    image_erode = cv2.erode(image, kernel, iterations= iterations)
    return image_erode

# Dilate image function
@st.cache
def img_dilate(image, iterations):
    kernel = np.ones((5, 5), np.uint8)
    image_dilate = cv2.dilate(image, kernel, iterations= iterations)
    return image_dilate

@st.cache
def img_gradient(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


# Canny Image
@st.cache
def to_canny(image, threshold1,threshold2):
    return cv2.Canny(to_gray(image), threshold1, threshold2)




#Gestion des modes de l'app
channel = 'RGB'
if mode == modes[0]:

    demo_image = st.sidebar.selectbox("Demo Image", options=demo_images)

    file = st.sidebar.file_uploader("Local Image", type = ['png', 'jpg', 'jpeg'])

    # Load Image
    image, channel = load_file(file,channel)
    
    left_column.markdown(f"<h2 style = 'text-align:center; color:red'>Original Image<h2/>", unsafe_allow_html=True)
    left_column.image(image, channels= channel)

    #Selectbox pour le traitemant de l'image

    process = st.sidebar.selectbox("Processing",
                                   ["Gray", "TrackColor", "Resize","Filter"])


    # Changer l'image en gray

    if process == "Gray":
        right_column.markdown(f"<h2 style = 'text-align:center; color:red'>Gray Image<h2/>", unsafe_allow_html=True)
        right_column.image(to_gray(image))
    # Redimentionner l'image
    if process == "Resize":
        right_column.markdown(f"<h2 style = 'text-align:center; color:red'>Resized Image<h2/>", unsafe_allow_html=True)
        st.sidebar.markdown("---")
        height = st.sidebar.slider("Heigth", min_value= 14,
                                   max_value= 4800, value= image.shape[0], step=2)
        width = st.sidebar.slider("Width", min_value=14, 
                                  max_value=4800, value=image.shape[1], step=2)
        right_column.image(img_resize(image, height= height, width=width),
                 channels= channel)

    # Threshold
    if process == "TrackColor":

        # Slider for the HSV values
        hue_min = st.sidebar.slider("Hue Min", min_value=0, max_value=179)
        hue_max = st.sidebar.slider("Hue Max", min_value=0, max_value=179, value=179)
        sat_min = st.sidebar.slider("Sat Min", min_value=0, max_value=255)
        sat_max = st.sidebar.slider("Sat Max", min_value=0, max_value=255,value= 255)
        val_min = st.sidebar.slider("Val Min", min_value=0, max_value=255)
        val_max = st.sidebar.slider("Val Max", min_value=0, max_value=255, value=255)

        # Convert the image to HSV image
        imgHsv = to_hsv(image)

        mask = mask(imgHsv, hue_min, sat_min, val_min, hue_max, sat_max, val_max)
        imgResult = color_mask(image, mask)
        right_column.markdown(f"<h2 style = 'text-align:center; color:red'>HSV Image<h2/>", unsafe_allow_html=True)
        right_column.image(imgHsv, channels = channel)
        left_column.markdown(f"<h2 style = 'text-align:center; color:red'>Mask<h2/>", unsafe_allow_html=True)
        left_column.image(mask)
        right_column.markdown(f"<h2 style = 'text-align:center; color:red'>Color Detected<h2/>", unsafe_allow_html=True)
        right_column.image(imgResult, channels = channel)
        

    # Filter process
    if process == "Filter":
        filters = ["Smooth", "Morphological", "Edge Detector"]
        filter = st.sidebar.selectbox("Type of filter", 
        options=filters)

        # Smooth Options
        if filter == filters[0]:
            blurs = ["Blur", "Median Blur", "Gaussian Blur"]
            blur = st.sidebar.selectbox("Type of Smooth",
            options=blurs)

            # Type blur
            if blur == blurs[0]:
                x = st.sidebar.slider("Blur kernel size", 1, 499, value=5, step=2)
                right_column.markdown(f"<h2 style = 'text-align:center; color:red'>Smooth Image<h2/>", unsafe_allow_html=True)
                right_column.image(blur_image(image, x), channels = channel)

            # Type Median Blur
            if blur == blurs[1]:
                x = st.sidebar.slider("Blur kernel size", 1, 361, value=5, step=2)
                right_column.markdown(f"<h2 style = 'text-align:center; color:red'>Smooth Image<h2/>", unsafe_allow_html=True)
                right_column.image(median_blur_image(image, x), channels = channel)

            # Type Gaussian
            if blur == blurs[2]:
                x = st.sidebar.slider("Blur kernel size", 1, 499, value=5, step=2)
                sigmaX = st.sidebar.slider("sigmaX", 0, 50, value=0, step=1)
                sigmaY = st.sidebar.slider("sigmaY", 0, 50, value=0, step=1)
                right_column.markdown(f"<h2 style = 'text-align:center; color:red'>Smooth Image<h2/>", unsafe_allow_html=True)
                right_column.image(gaussian_blur_image(image, x, sigmaX, sigmaY), channels = channel)

        # Morphological filter
        if filter == filters[1]:
            morphology = ["Erosion", "Dilation", "Gradient"]
            morphology_type =st.sidebar.selectbox("Type", options=morphology)
            
            # Erosion
            if morphology_type == morphology[0]:
                
                iterations = st.sidebar.slider("Iteration", min_value=1, max_value=20, value=1)
                image_erode = img_erosion(to_gray(image), iterations)
                right_column.markdown(f"<h2 style = 'text-align:center; color:red'>Erode Image<h2/>", unsafe_allow_html=True)
                right_column.image(image_erode)

            
            # Dilation
            if morphology_type == morphology[1]:
                threshold1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
                threshold2 = st.sidebar.slider("Threshold 2", 0, 255, 120)
                iterations = st.sidebar.slider("Iteration", min_value=1, max_value=20, value=1)
                image_dilate = img_dilate(to_canny(image, threshold1, threshold2), iterations)
                right_column.markdown(f"<h2 style = 'text-align:center; color:red'>Dilate Image<h2/>", unsafe_allow_html=True)
                right_column.image(image_dilate)

            # Gradient
            if morphology_type == morphology[2]:
                threshold1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
                threshold2 = st.sidebar.slider("Threshold 2", 0, 255, 120)
                gradient = img_gradient(to_canny(image, threshold1, threshold2))
                right_column.markdown(f"<h2 style = 'text-align:center; color:red'>Gradient Image<h2/>", unsafe_allow_html=True)
                right_column.image(gradient)

        if filter == filters[2]:
            threshold1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
            threshold2 = st.sidebar.slider("Threshold 2", 0, 255, 120)
            edge = cv2.Canny(to_gray(image),threshold1,threshold2)
            right_column.markdown(f"<h2 style = 'text-align:center; color:red'>Edge Detector<h2/>", unsafe_allow_html=True)
            #print(edge.shape)
            right_column.image(edge)




if mode == modes[1] :

    # Load Video
    demo_file = st.sidebar.selectbox("Demo Video", options=demo_videos)
    video_file = st.sidebar.file_uploader("Video", type =["avi", "mp4"])



    # Checkbox to use the camera
    use_webcam = st.sidebar.checkbox("camera")
    html_markdown("Output Video")
    #st.markdown("<hr/>", unsafe_allow_html= True)

    tfile = tempfile.NamedTemporaryFile(delete=False)

    # Choise video file
    if not video_file:
        if use_webcam:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(demo_file)
            tfile.name = demo_file
    
    else:
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("out.mp4", codec, fps_input, (width, height))

    face = st.sidebar.selectbox("Face", options = ["Run Video","Face Mesh", 'Face Detector'])
    sempty = st.sidebar.empty()
    max_face_placeholder = st.sidebar.empty()

    # Show the video in the sidebar and the main page
    if not use_webcam:
        st.sidebar.text("Input Video")
        st.sidebar.video(tfile.name)

    draw_spec1 = mp_drawing.DrawingSpec(thickness=1, circle_radius = 1, color= (0, 255, 0))
    draw_spec2 = mp_drawing.DrawingSpec(thickness=1, circle_radius = 1, color= (255, 0, 0))

    sf = st.empty()
    if face=="Run Video" and not use_webcam:
        sf.video(tfile.name)

    if face == "Face Mesh":
        
        # Record the video
        record = sempty.checkbox("Record")

        # Choise the max number of face to detect

        max_face = max_face_placeholder.number_input("Maximun Face detected",1, 5, 1)
        left , midle, right, last = st.columns(4)
        left.markdown("<h4 style = 'text-align:center; color:blue'>Frame Per Second<h4/>", unsafe_allow_html=True)
        midle.markdown("<h4 style = 'text-align:center; color:blue'>Face Detected<h4/>", unsafe_allow_html=True)
        right.markdown("<h4 style = 'text-align:center; color:blue'>Video Width<h4/>", unsafe_allow_html=True)
        last.markdown("<h4 style = 'text-align:center; color:blue'>Video Height<h4/>", unsafe_allow_html=True)

        # Video Descriptions placeholders

        k1 = left.empty()
        k2 = midle.empty()
        k3 = right.empty()
        k4 = last.empty()

        k1.markdown("0")
        k2.markdown("0")
        k3.markdown("0")
        k4.markdown("0")


        # Face Mesh code
        with mp_face_mesh.FaceMesh(max_num_faces = max_face) as face_mesh:

            cTime = 0
            pTime = 0

            # Video Loop
            while cap.isOpened():

                success, frame = cap.read()
                
                if not success:
                    break
                
                #frame = img_resize(frame, 640, 480)
                #print(frame.shape)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(frame)
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                face_count = 0
                if result.multi_face_landmarks:
                    
                    for facelms in result.multi_face_landmarks:
                        mp_drawing.draw_landmarks(frame, facelms, mp_face_mesh.FACE_CONNECTIONS, draw_spec1, draw_spec2)
                        face_count += 1

                if record:
                    out.write(frame)
                cTime = time.time()
                fps = int(1/(cTime-pTime))
                k1.markdown(f"<h1 style = 'text-align:center; color:red'>{int(fps)}<h1/>", unsafe_allow_html=True)
                k2.markdown(f"<h1 style = 'text-align:center; color:red'>{face_count}<h1/>", unsafe_allow_html=True)
                k3.markdown(f"<h1 style = 'text-align:center; color:red'>{width}<h1/>", unsafe_allow_html=True)
                k4.markdown(f"<h1 style = 'text-align:center; color:red'>{height}<h1/>", unsafe_allow_html=True)


                pTime = cTime
                #frame = cv2.resize(frame,(0, 0), fx =0.8, fy = 0.8)
                #frame = img_resize(frame, width = 640)
                sf.image(frame, channels= 'BGR', use_column_width= True)   
    cap.release()
    out.release()
    cv2.destroyAllWindows()


st.markdown("<hr/>", unsafe_allow_html= True)

with st.expander("About Me" ):
    st.markdown("""<body><div>My name is<div/>
                <h2 style='color:blue'>SOUSSOU Desire<h2/>
                <div>I'am a student at Lome Universty of Togo
                This is my first project with Streamlit<div/> <body/>""", unsafe_allow_html=True)
    st.write("""My Contact""")
    st.markdown("""<h3 style = 'color:blue'>+22879627154<h3/>""", unsafe_allow_html=True)