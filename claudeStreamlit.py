import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2



def main():
    st.title('Crack Detection Live Stream')
    
    # Load the YOLO model
    model = YOLO('best.pt')
    
    # Start/Stop detection
    run_detection = st.checkbox('Start Crack Detections')
    
    if run_detection:
        # Create a placeholder for the video stream
        video_placeholder = st.empty()
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        while run_detection:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                st.error('Failed to grab frame ')
                break
            
            # Perform prediction on the frame
            results = model.predict(source=frame, save=False, show_boxes=True, show=False)
            
            # Process each result
            for result in results:
                # Annotate the frame with detections
                annotated_frame = result.plot()
                
                # Convert from BGR to RGB for Streamlit
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display the annotated frame
                video_placeholder.image(annotated_frame_rgb, channels="RGB")
            
        # Release the capture when detection stops
        cap.release()
    
    # Optional: Display class information
    st.sidebar.header('Detection Information')
    try:
        st.sidebar.write('Available Classes:', model.names)
    except:
        st.sidebar.write('Could not retrieve class information')

if __name__ == '__main__':
    main()
