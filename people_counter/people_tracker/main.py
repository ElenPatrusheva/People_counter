"""
Track people using centroids

Attributes:
    DETECT_FREQ (int): Description
    DISTANCE_THRESH (int): maximum eucledian distance
    MAX_MISSES (int): number unmatched detections before tracker deletion
    MIN_HITS (int): number matched detections before tracker creation
    next_id (int): next available id
    persons_list (list): list of person trackers
    SAVE_VIDEO (bool): set as True if you need to save the video
    SHOW_SCALE (float): scale of the displayed video
"""
import numpy as np
import cv2
import sys
from scipy.optimize import linear_sum_assignment
import time
import argparse
import tools
import detector
import detector_yolo
import tracker_cv2 as trc
#import tracker_dlib as trc
from imutils.video import FPS



# Whether to save output video or no
SAVE_VIDEO = True

# Number of consecutive unmatched detections before tracker deletion
MAX_MISSES = 10

# Number of consecutive matched detections before tracker creation
MIN_HITS = 2

# Output video scale
SHOW_SCALE = 0.5

# Maximum eucledian distance between
# detection and tracker to be considered as a match
DISTANCE_THRESH = 100


persons_list = []
next_id = 0


def assign_detections_to_trackers(trackers, detections, dst_thrd=100):
    """
    Solves data association problem

    Args:
        trackers (list): list of trackers' bounding boxes
        detections (list): list of detections' bounding boxes
        dst_thrd (float, optional): maximum distance btw matched box centers

    Returns:
        (list, list, list): matches, new detections, unmatched trackers
    """

    # Create cost matrix
    eucledian_costs = np.zeros(
        (len(trackers), len(detections)), dtype=np.float32)

    # Fill cost matrix using Eucledian Distance metric
    for t, trk in enumerate(trackers):
        for d, det in enumerate(detections):
            detection = [det[0] + int((det[2] - det[0]) / 2),
                         det[1] + int((det[3] - det[1]) / 2)]
            tracker = [trk[0] + int((trk[2] - trk[0]) / 2),
                       trk[1] + int((trk[3] - trk[1]) / 2)]
            eucledian_costs[t, d] = tools.eucledian(tracker, detection)

    # Solve data association problem
    matched_idx = linear_sum_assignment(eucledian_costs)
    matched_idx = np.asanyarray(matched_idx).T

    unmatched_trackers, unmatched_detections = [], []

    # Save trackers without new detections
    for t, trk in enumerate(trackers):
        if(t not in matched_idx[:, 0]):
            unmatched_trackers.append(t)

    # Save new detections that have no tracker
    for d, det in enumerate(detections):
        if(d not in matched_idx[:, 1]):
            unmatched_detections.append(d)

    matches = []

    # Check the "safe" distance between detection and tracker to be less
    # than dst_trhd. Append tracker to unmatched trackers otherwise
    for m in matched_idx:
        if(eucledian_costs[m[0], m[1]] > dst_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    # Prepare matches
    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def track(img):
    """
    Tracks people without detection

    Args:
        img (np.array): current frame
        start_time (float): time when the current frame was captured

    Returns:
        np: current frame with bounding boxes
    """
    print('[TRACKING]')
    global persons_list

    # Predict state for each tracker without measurement
    # TODO what to do if the box disappeared
    for person in persons_list:
        person.update_state(img)

        x = int((person.box[0] + person.box[2])/2)
        y = int((person.box[1] + person.box[3])/2)
        center = (y, x) 
        img = cv2.circle(img, center, 7, (0, 255, 0), -1) 
        img = tools.draw_box(person.id, img, person.box)

    return img


def detect(img, start_time, det):
    """
    Tracks people with detection

    Args:
        img (np.array): current frame
        start_time (float): time when the current frame was captured

    Returns:
        np: current frame with bounding boxes
    """
    print('[DETECTION]')
    global persons_list
    global next_id

    # Get detections
    detections = det.get_detections(img)

    # Create list of existing persons' boxes
    persons_boxes = []
    if len(persons_list) > 0:
        for person in persons_list:
            persons_boxes.append(person.box)

    # Solve data association problem
    matched, unmatched_dets, unmatched_trks \
        = assign_detections_to_trackers(persons_boxes, detections, dst_thrd=DISTANCE_THRESH)

    # Update trackers with measurements
    if matched.size > 0:
        for person_idx, det_idx in matched:
            # Extract measurement
            z = detections[det_idx]
            center = [z[0] + int((z[2] - z[0]) / 2),
                      z[1] + int((z[3] - z[1]) / 2)]
            
            img = cv2.circle(img, (center[1], center[0]), 6, (0, 0, 255), -1)
            
            person.hits += 1
            person.misses = 0
            person.box = tuple(z) # NEW

    # Update trackers without measurements

    # TODO check removed unmatched trackers
    for person_idx in unmatched_trks:
        person = persons_list[person_idx]
        person.misses += 1
        person.hits = 0
        

    # Create tracker for new detections
 
    for idx in unmatched_dets:
        z = detections[idx]
        person = trc.Tracker(tuple(z), img)
        #person.box = z

        # Assign an ID for the tracker
        next_id += 1
        person.id = next_id

        persons_list.append(person)
        persons_boxes.append(z)

    # Draw good boxes
    for person in persons_list:
        if ((person.hits >= MIN_HITS) and (person.misses <= MAX_MISSES)):
            x = int((person.box[1] + person.box[3])/2)
            y = int((person.box[0] + person.box[2])/2)

            img = cv2.circle(img, (y, x), 4, (0, 255, 0), -1)
            img = tools.draw_box(person.id, img, person.box)
    # Remove lost trackers
    persons_list = [i for i in persons_list if i.misses <= MAX_MISSES]

    return img


def main(args):
    """
    Processing video, detecting and tracking people on it 
    """
    
    det = detector_yolo.Detector()

    cap = cv2.VideoCapture(args['input'])
    # cap = cv2.VideoCapture(0)

    if (SAVE_VIDEO):
        video_width = int(cap.get(3))
        video_height = int(cap.get(4))
        video_fps = int(cap.get(5))
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(args['output'], fourcc, video_fps,
            (video_width, video_height), True)

    frame = 0
    start_time = time.time()

    while(True):
        _, img = cap.read()

        if (img is None):
            break

        if (frame % args['skip_frames'] == 0):
            print("[Tracking]")
            result = detect(img, start_time, det)
        else:
            print("[Predicting]")
            result = track(img)
        # Save frame
        '''if (SAVE_VIDEO):
            out.write(result)'''

        frame += 1

        result = cv2.resize(result, (0, 0), fx=SHOW_SCALE, fy=SHOW_SCALE)
        cv2.putText(result, f'FPS: {frame/(time.time()-start_time)}', (0, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
        cv2.imshow("frame", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if (SAVE_VIDEO):
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    '''ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
        help="minimum probability to filter weak detections")'''

    ap.add_argument("-i", "--input", type=str, required=True,
        help="path to input video file")
    ap.add_argument("-o", "--output", type=str, required=True,
        help="path to output video file")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
        help="# of skip frames between detections")
    ap.add_argument("-p", '--polygones', type=str, 
        help="Path to the file with list of polygones")

    args = vars(ap.parse_args())
    main(args)
cxdx