"""
CV2 filter for tracking people
"""
import cv2

class Tracker():

    """
    Kalman filter class. Tracks people with help of detections as measurements

    Attributes:
        box (list): coordinates of a bounding box
        hits (int): number of detection matches
        id (int): person ID
        misses (int): number of detection mismatches
    """

    def __init__(self, box, frame):
        """
        Initializes Kalman filter
        """
        # Person ID
        self.id = 0

        # Coordinates of a bounding box
        self.box = box

        # Number of detection matches
        self.hits = 1

        # Number of detection mismatches
        self.misses = 0

        self.tracker = cv2.TrackerMOSSE_create()#fastest still 5 fps

        #self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, box)


    def update_state(self, frame):
        """
        Update tracker
        """
        ok, box = self.tracker.update(frame)
        if ok:
            box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            self.box = box