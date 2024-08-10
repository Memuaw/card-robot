import cv2
import numpy as np
import glob

# Define global variables
images = [cv2.imread(file) for file in glob.glob("input/Card Detector/Build Dataset/Card Images/*.jpg")]

convex_hulls = []

# Algorithm for gathering the convex hulls
def gather():
    # Function to get a single convex hull for a given region
    def get_combined_convex_hull(image, br):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply canny filtering
        edges = cv2.Canny(gray, 50, 150)
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine all filtered contours into a single set of points
        all_points = np.concatenate(contours)
        
        if br == True:
            points = []
            h, w, _ = image.shape

            for point in all_points:
                points.append([point[0][0] + w * 6.8, point[0][1] + h * 2.6])
    
            points_array = np.array(points, dtype=np.float32)
            
            hull = cv2.convexHull(points_array)

            return hull

        # Get the convex hull for the combined points
        hull = cv2.convexHull(all_points)
        return hull
    
    for image in images:
        height, width, _ = image.shape
        # Define ROIs (Top-left and Bottom-right corners)
        roi_tl = image[int(height*0.025):int(height*0.3), int(width*0.025):int(width*0.15)]  
        roi_br = image[int(height*0.7):int(height*0.975), int(width*0.85):int(width*0.975)]

        # Get combined convex hulls for the ROIs
        hull_tl = get_combined_convex_hull(roi_tl, False)
        hull_br = get_combined_convex_hull(roi_br, True)
        convex_hulls.append([hull_tl, hull_br])
    
    return convex_hulls

convex_hull_list = gather()