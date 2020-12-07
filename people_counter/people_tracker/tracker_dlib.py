"""
dlib filter for tracking people
"""
import dlib

class Tracker():

    """
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
        self.height = frame.shape[0]
        self.width = frame.shape[1]
        print(self.id, box)
        # Coordinates of a bounding box
        self.box = box
        if box[0]<0 or box[1]<0 or box[2]<0 or box[3]<0:
            print('---------------------')
            print('Unexpected box', box)
            print('---------------------')

        # Number of detection matches
        self.hits = 1

        # Number of detection mismatches
        self.misses = 0

        self.tracker = dlib.correlation_tracker()
        cent_x = int((box[0] + box[2])/2)
        cent_y = int((box[1] + box[3])/2)
        w = box[3] - box[1]
        h = box[2] - box[0]
        rect = dlib.centered_rect(dlib.point(cent_x, cent_y), h, w) 
        self.tracker.start_track(frame, rect)

    def update_state(self, frame):
        """
        Update tracker
        """
        self.tracker.update(frame)
        pos = self.tracker.get_position()
        # unpack the position object
        left = int(pos.left())
        top = int(pos.top())
        right = int(pos.right())
        bottom = int(pos.bottom())

        self.box = (left,top,right,bottom)
