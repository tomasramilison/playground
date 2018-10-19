import cv2, time, numpy as np

# test pictures
pic1 = 'test_images/i_robot_1.jpg'  # robot body image
pic2 = 'test_images/i_robot_2.jpg'  # robot full body image
pic3 = 'test_images/i_robot_3.jpg'  # robot face image
pic4 = 'test_images/human_face.jpg'  # human face
pic5 = 'test_images/human_upper_body.jpg'  # human upper body
pic6 = 'test_images/full_body_color_1.jpg'  # human full body in suit
pic7 = 'test_images/full_body_color_2.jpg'  # human full body casual clothes
# pic8 = 'test_images/pedestrians_1.jpg'      # 4 humans (full body) WARNING: DO NOT USE ON CPU
pic9 = 'test_images/stock_pic_man.jpg'  # human arms up full body
pic10 = 'test_images/fullbody-copy.jpg'  # human in suit full body 1
pic11 = 'test_images/fullbody_x.jpg'  # human in suit full body 2
pic12 = 'test_images/full_body_y1.jpg'  # human full body
pic13 = 'test_images/spiderman_full_1.jpg'  # spiderman full
pic14 = 'test_images/human_full_y.jpg'  # full body
pic15 = 'test_images/test_15.jpg'  # human arms up
pic16 = 'test_images/test_16.jpg'  # human arm up (R)
pic17 = 'test_images/test_17.jpg'  # human arm up (L)
pic18 = 'test_images/test_18.jpg'  # human arms on hips

# cascade classifiers
cc_front = cv2.CascadeClassifier('pretrained/haarcascade_frontalface_default.xml')
cc_profile = cv2.CascadeClassifier('pretrained/haarcascade_profileface.xml')
# cc_full = cv2.CascadeClassifier('pretrained/haarcascade_fullbody.xml')
cc_upper = cv2.CascadeClassifier('pretrained/HS.xml')


def estimate_skin_tone(face_roi):
    """ Find average color in region of image corresponding to face

        :param face_roi: region of interest where face is detected as matrix
        :return: RGB value of skin tone estimate
    """
    return [int(face_roi[:, :, i].mean()) for i in range(face_roi.shape[-1])]


def video_test():
    """ TEST: Image processing in real-time (face, upper body and full body detection) """
    # video capture
    cap = cv2.VideoCapture(0)
    while True:
        # read picture
        _, frame = cap.read()
        # gray = cv2.cvtColor(src=frame, code=cv2.COLOR_RGBA2GRAY)
        '''
        # detect face, upper body and full body
        face, _01, _02 = cc_front.detectMultiScale3(image=gray, scaleFactor=1.3,
                                                    minNeighbors=5, outputRejectLevels=True)
        upper_body, _10, _20 = cc_upper.detectMultiScale3(image=gray, scaleFactor=1.1,
                                                          minNeighbors=7, outputRejectLevels=True)
        full_body, _001, _002 = cc_full.detectMultiScale3(image=gray, scaleFactor=1.008,
                                                          minNeighbors=6, outputRejectLevels=True)
        # draw rectangles
        for (x, y, w, h) in face:
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
        for (x, y, w, h) in upper_body:
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 0), thickness=2)
        for (x, y, w, h) in full_body:
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
        '''
        frame = limb_tracker(frame=frame)
        # display the resulting frame
        cv2.imshow(winname='Frame', mat=frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # when everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def centroid_of_rect(roi):
    """ Finds the coordinates for the centroid of a rectangle.

        :param roi: rectangle region of interest as matrix
        :return: coordinates of centroid within region of interest
    """
    return int(roi.shape[0] / 2), int(roi.shape[1] / 2)


def approx_shoulders(upper_body_roi):
    """ Approximate the coordinates of shoulders based on upper body region of interest detection.
        Here I assume that these coordinates are more often than not located as follows:

            -> Right shoulder: 1/4 of height on Y-axis, 1/6 of width on X-axis
            -> Left shoulder: 1/4 of height on Y-axis, 5/6 of width on X-axis

        :param upper_body_roi: rectangle region of interest as matrix
        :return: 2 sets of coordinates for shoulder positions
    """
    height = upper_body_roi.shape[0]; width = upper_body_roi.shape[1]
    return (int(width / 6), int((height / 4) * 3)), (int((width / 6) * 5), int((height / 4) * 3))


def approx_biceps(thresh, shoulder_pts, dst):
    """ Approximate biceps position. This is achieved by creating an array of line segments all
        starting at the shoulder points, going outwards at inclinations of n degrees (n being every
        integer value between 1) 10° up 180°, and 2) 350° down 180°, for the right and left arm
        respectively). Once all the lines are found, we can iterate through all the end points of
        these lines to determine whether they are white or black in 'thresholded' frame.

        NOTE: If threshold op is performed on hue channel using cv2.THRESH_BINARY & cv2.THRESH_OTSU
        threshold types, it will likely yield a result where the human shape is rendered white and
        background pixels black.

        :param thresh: 'thresholded' image as matrix
        :param shoulder_pts: tuple of approximated shoulder positions
        :param dst: normal frame as matrix
        :return: tuple of approximated coordinates corresponding to the tip of biceps
    """
    biceps_R = None; biceps_L = None

    # r_lines = []; l_lines = []  # arrays of lines for right/left arm

    #############
    # RIGHT ARM # FIXME: This is actually the left arm
    #############
    # starting point is the right shoulder coordinates
    r_start1 = shoulder_pts[1]
    for i in range(10, 180):
        ''' We first determine a segment's delimiting point by doing the following:

                -> x2 = (x1 + (sin(i° * PI / 180°) * 142px))
                -> y2 = (y1 + (cos(i° * PI / 180°) * 142px))

            This (x2, y2) gives us the coordinates of the end point of a straight line
            segment starting at point (x1, y1) going outward with an inclination of i°
            with respect to 180°. The 142px value is arbitrary, it's the measure that 
            happens to work best with most of my test photos.
        '''
        sin = np.sin(i * np.pi / 180); cos = np.cos(i * np.pi / 180)
        r_end1 = (int(r_start1[0] + (sin * 142)), int(r_start1[1] + (cos * 142)))
        try:
            ''' NOTE: In cv2, the pixel lookup method seems to be inverted(?); coordinates in
                matrix are encoded as (y, x) instead of (x, y), hence the need to invert the
                order of r_end1 tuple in statement below. I'm unsure why it is this way but it
                took an embarrassing amount of time to figure it out. '''
            if thresh[r_end1[1], r_end1[0]] == 255:
                cv2.circle(img=dst, center=r_end1, radius=3, color=(0, 0, 255), thickness=8)
                biceps_R = r_end1
                ''' Break from loop once we've found the first point starting from the bottom.
                    ALTERNATIVELY: We may store all the white points in a list (comment below)
                    and later loop through all of them to determine which one is closest/farthest
                    from shoulder point by using a distance metric like Hamming or Euclidean. '''
                # r_lines.append(r_end1)
                break
        except IndexError:  # catch IndexError for smaller frames
            continue

    ############
    # LEFT ARM #
    ############
    # starting point is the left shoulder coordinates
    l_start1 = shoulder_pts[0]
    ''' Same as previously only decrementing from 350° to 180°. '''
    for i in range(350, 180, -1):
        sin = np.sin(i * np.pi / 180); cos = np.cos(i * np.pi / 180)
        l_end1 = (int(l_start1[0] + (sin * 142)), int(l_start1[1] + (cos * 142)))
        try:
            if thresh[l_end1[1], l_end1[0]] == 255:
                cv2.circle(img=dst, center=l_end1, radius=3, color=(0, 0, 255), thickness=8)
                biceps_L = l_end1
                # l_lines.append(l_end1)
                break
        except IndexError:
            continue
    return biceps_L, biceps_R


def approx_forearms(shoulder_pts, biceps_pts, thresh, dst):
    """ Approximate forearms position. This is achieved in a way similar to the biceps detection
        system; here, we use a measure of the angle of inclination of the biceps to determine where
        and how to execute our white pixel probe.

    :param shoulder_pts: shoulder coordinates
    :param biceps_pts: biceps coordinates
    :param thresh: 'thresholded' frame as a matrix
    :param dst: normal frame as matrix
    :return: tuple of coordinates corresponding to the approx. tip of forearms
    """
    forearm_R = None; forearm_L = None

    #############
    # RIGHT ARM # FIXME: This is actually the left arm
    #############
    r_start = biceps_pts[1]
    try:
        ''' Given the equation of a line, y = mx + b, we can find the inclination like so:

                -> Slope: m = (y2 - y1) / (x2 - x1)
                -> Angle: theta = tan^-1(slope)
        '''
        r_shoulder = shoulder_pts[1]
        r_slope = (r_start[1] - r_shoulder[1]) / (r_start[0] - r_shoulder[0])
        r_incl = np.arctan(r_slope)
    except TypeError:
        r_incl = 0
    if 1 > abs(r_incl) > 0.01:
        for i in range(210, -90, -1):
            sin = np.sin(i * np.pi / 180); cos = np.cos(i * np.pi / 180)
            try:
                r_end = (int(r_start[0] + (sin * 128)), int(r_start[1] + (cos * 128)))
            except TypeError:
                continue
            try:
                if thresh[r_end[1], r_end[0]] == 255:
                    cv2.circle(img=dst, center=r_end, radius=3, color=(0, 0, 255), thickness=8)
                    forearm_R = r_end
                    break
            except IndexError:
                continue
    else:
        for i in range(0, 180):
            sin = np.sin(i * np.pi / 180); cos = np.cos(i * np.pi / 180)
            try:
                r_end = (int(r_start[0] + (sin * 128)), int(r_start[1] + (cos * 128)))
            except TypeError:
                continue
            try:
                if thresh[r_end[1], r_end[0]] == 255:
                    cv2.circle(img=dst, center=r_end, radius=3, color=(0, 0, 255), thickness=8)
                    forearm_R = r_end
                    break
            except IndexError:
                continue
    ############
    # LEFT ARM #
    ############
    l_start = biceps_pts[0]
    try:
        l_shoulder = shoulder_pts[0]
        l_slope = (l_start[1] - l_shoulder[1]) / (l_start[0] - l_shoulder[0])
        l_incl = np.arctan(l_slope)
    except TypeError:
        l_incl = 0
    if 1 > abs(l_incl) > 0.01:
        for i in range(210, 90, -1):
            sin = np.sin(i * np.pi / 180); cos = np.cos(i * np.pi / 180)
            try:
                l_end = (int(l_start[0] + (sin * 128)), int(l_start[1] + (cos * 128)))
            except TypeError:
                continue
            try:
                if thresh[l_end[1], l_end[0]] == 255:
                    cv2.circle(img=dst, center=l_end, radius=3, color=(0, 0, 255), thickness=8)
                    forearm_L = l_end
                    break
            except IndexError:
                continue
    else:
        for i in range(360, 180, -1):
            sin = np.sin(i * np.pi / 180); cos = np.cos(i * np.pi / 180)
            try:
                l_end = (int(l_start[0] + (sin * 128)), int(l_start[1] + (cos * 128)))
            except TypeError:
                continue
            try:
                if thresh[l_end[1], l_end[0]] == 255:
                    cv2.circle(img=dst, center=l_end, radius=3, color=(0, 0, 255), thickness=8)
                    forearm_L = l_end
                    break
            except IndexError:
                continue
    return forearm_L, forearm_R


def limb_tracker(frame=None, path=None, memory_efficient=True):
    """ Track humanoid limbs in an image.

        :param path: path to image
        :param frame: frame as matrix
        :param memory_efficient: True compromises accuracy for speed of execution
        :return frame with tracked limbs as matrix
    """
    start = time.time()
    # initialize matrix
    if path is not None:
        frame = cv2.imread(filename=path)
    # operations on frame
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_RGBA2GRAY)
    # remove noise from frame
    frame = cv2.fastNlMeansDenoisingColored(src=frame)
    # create hsv and split channels
    hsv = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    # create 'thresholded'
    _, thresholded = cv2.threshold(src=hue, thresh=0, maxval=255,
                                   type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # detect face, hands, upper body and full body
    face, _, _ = cc_front.detectMultiScale3(image=gray, scaleFactor=1.3, minNeighbors=5,
                                            outputRejectLevels=True)

    face_roi = None; face_roi_in_frame = None; shoulders = None
    if not memory_efficient:
        upper_body, _, _ = cc_upper.detectMultiScale3(image=gray, scaleFactor=1.1, minNeighbors=7,
                                                      outputRejectLevels=True)
        upper_body = []
        upper_roi = None

        ##################
        # FACE DETECTION #
        ##################
        for (x, y, w, h) in face:
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
            face_roi = frame[y: y + h, x: x + w]
        try:
            # first match template in original frame
            ''' The matchTemplate() method requires three parameters. The first is the full 
                frame image (i.e. the image that contains what we are searching for), the second
                is our query image (i.e. the image for which we're trying to pinpoint the location)
                and the third is the matching method we wish to use (there are a number of methods
                to perform template matching, but in this case I am using the 'correlation 
                coefficient' which is specified by the flag cv2.TM_CCOEFF). '''
            result = cv2.matchTemplate(image=frame, templ=face_roi, method=cv2.TM_CCOEFF)
            # here I get min/max location values (only max will be used, that's the top left corner)
            _, _, minloc, maxloc = cv2.minMaxLoc(src=result)
            topleft = maxloc
            # grab the bounding box of roi and extract it from the frame image
            botright = (topleft[0] + face_roi.shape[0], topleft[1] + face_roi.shape[1])
            face_roi_in_frame = frame[topleft[1]:botright[1], topleft[0]:botright[0]]
        except (AttributeError, cv2.error):
            print('Error 001 occurred')
            pass
        ########################
        # UPPER BODY DETECTION #
        ########################
        for (x, y, w, h) in upper_body:
            # cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255),
            #               thickness=2)
            upper_roi = frame[y: y + h, x: x + w]
            upper_body_surface = (y + h) * (x + w)
            print('Centroid upper body pixel: ' + str(centroid_of_rect(roi=upper_roi)))
            shoulders = approx_shoulders(upper_body_roi=upper_roi)
            try:
                cv2.circle(img=upper_roi, center=shoulders[0], color=(0, 0, 255),
                           radius=3, thickness=8)
                cv2.circle(img=upper_roi, center=shoulders[1], color=(0, 0, 255),
                           radius=3, thickness=8)
            except (AttributeError, cv2.error):
                pass
        try:
            result = cv2.matchTemplate(image=frame, templ=upper_roi, method=cv2.TM_CCOEFF)
            _, _, minloc, maxloc = cv2.minMaxLoc(src=result)
            topleft = maxloc
            botright = (topleft[0] + upper_roi.shape[0], topleft[1] + upper_roi.shape[1])
            upper_roi_in_frame = frame[topleft[1]:botright[1], topleft[0]:botright[0]]
            cv2.circle(img=upper_roi_in_frame, center=centroid_of_rect(roi=upper_roi_in_frame),
                       color=(0, 0, 255), radius=3, thickness=8)
            shoulders_in_frame = ((maxloc[0] + shoulders[0][0], maxloc[1] + shoulders[0][1]),
                                  (maxloc[0] + shoulders[1][0], maxloc[1] + shoulders[1][1]))
            b_arms = approx_biceps(thresh=thresholded, shoulder_pts=shoulders_in_frame, dst=frame)
            print('Left biceps coordinates: ' + str(b_arms[1]))
            print('Right biceps coordinates: ' + str(b_arms[0]))
            # point between shoulders
            mid_shoulders_pt = centroid_of_rect(roi=upper_roi_in_frame)[0], shoulders[0][1]
            cv2.circle(img=upper_roi, center=mid_shoulders_pt, radius=3, color=(0, 0, 255),
                       thickness=8)
            f_arms = approx_forearms(shoulder_pts=shoulders_in_frame, biceps_pts=b_arms,
                                     thresh=thresholded, dst=frame)
            print('Left forearm coordinates: ' + str(f_arms[1]))
            print('Right forearm coordinates: ' + str(f_arms[0]))

            ''' Draw lines on frame for points we have so far '''

            if shoulders_in_frame[0] is not None and b_arms[0] is not None:
                #  left shoulder joint to biceps
                cv2.line(img=frame, pt1=shoulders_in_frame[0], pt2=b_arms[0],
                         color=(255, 0, 128), thickness=3)
            if shoulders_in_frame[1] is not None and b_arms[1] is not None:
                # right shoulder joint to biceps
                cv2.line(img=frame, pt1=shoulders_in_frame[1], pt2=b_arms[1],
                         color=(255, 0, 128), thickness=3)
            if b_arms[0] is not None and f_arms[0] is not None:
                # right biceps joint to forearm
                cv2.line(img=frame, pt1=b_arms[0], pt2=f_arms[0], color=(255, 0, 128), thickness=3)
            if b_arms[1] is not None and f_arms[1] is not None:
                # left biceps joint to forearm
                cv2.line(img=frame, pt1=b_arms[1], pt2=f_arms[1], color=(255, 0, 128), thickness=3)
            # join shoulders and mid-shoulders point
            cv2.line(img=upper_roi_in_frame, pt1=shoulders[0], pt2=mid_shoulders_pt,
                     color=(255, 0, 128), thickness=3)
            cv2.line(img=upper_roi_in_frame, pt1=shoulders[1], pt2=mid_shoulders_pt,
                     color=(255, 0, 128), thickness=3)
            # mid-shoulders point to neck joint
            cv2.line(img=upper_roi_in_frame, pt1=mid_shoulders_pt,
                     pt2=centroid_of_rect(roi=upper_roi_in_frame), color=(255, 0, 128), thickness=3)
        except (AttributeError, cv2.error):
            print('Error 002 occurred')
            pass

        # track time
        end = time.time(); exectime = end - start
        print('Execution time for initial detect: ' + str(exectime) + ' secs')

        return frame

    else:
        ##################
        # FACE DETECTION #
        ##################
        for (x, y, w, h) in face:
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
            face_roi = frame[y: y + h, x: x + w]
        try:
            result = cv2.matchTemplate(image=frame, templ=face_roi, method=cv2.TM_CCOEFF)
            _, _, minloc, maxloc = cv2.minMaxLoc(src=result)
            topleft = maxloc
            botright = (topleft[0] + face_roi.shape[0], topleft[1] + face_roi.shape[1])
            face_roi_in_frame = frame[topleft[1]:botright[1], topleft[0]:botright[0]]
            face_bottom = (maxloc[0] + int(face_roi_in_frame.shape[0]/2),
                           maxloc[1] + face_roi_in_frame.shape[1])
            cv2.circle(img=frame, center=face_bottom, radius=3, color=(0, 0, 255), thickness=8)
            ''' Approximate mid-shoulders point to the same value on X-Axis as face bottom, plus 
                height of face rectangle, minus 1/5th of that same value. '''
            mid_shoulders_pt = (face_bottom[0],
                                face_bottom[1] +
                                face_roi_in_frame.shape[1] -
                                int(face_roi_in_frame.shape[1]/5))
            cv2.circle(img=frame, center=mid_shoulders_pt, radius=3, color=(0, 0, 255), thickness=8)
            ''' Approximate right shoulder point to the same value on X-Axis as mid-shoulders, minus
                width of face rectangle, minus 1/15th of that same value.'''
            r_shoulder = (mid_shoulders_pt[0] -
                          face_roi_in_frame.shape[0] -
                          int(face_roi_in_frame.shape[0]/15),
                          mid_shoulders_pt[1])
            cv2.circle(img=frame, center=r_shoulder, radius=3, color=(0, 0, 255), thickness=8)
            ''' Approximate left shoulder point to the same value on X-Axis as mid-shoulders, plus
                width of face rectangle, plus 1/15th of that same value.'''
            l_shoulder = (mid_shoulders_pt[0] +
                          face_roi_in_frame.shape[0] +
                          int(face_roi_in_frame.shape[0] / 4),
                          mid_shoulders_pt[1])
            cv2.circle(img=frame, center=l_shoulder, radius=3, color=(0, 0, 255), thickness=8)
            shoulders = (r_shoulder, l_shoulder)
            biceps = approx_biceps(thresh=thresholded, shoulder_pts=shoulders, dst=frame)
            forearms = approx_forearms(shoulder_pts=shoulders, biceps_pts=biceps,
                                       thresh=thresholded, dst=frame)

            ''' Draw lines for points obtained '''
            # mid-shoulders to face bottom
            cv2.line(img=frame, pt1=mid_shoulders_pt, pt2=face_bottom,
                     color=(255, 0, 128), thickness=3)
            if shoulders[0] is not None and biceps[0] is not None:
                #  left shoulder joint to biceps
                cv2.line(img=frame, pt1=shoulders[0], pt2=biceps[0],
                         color=(255, 0, 128), thickness=3)
                # left shoulder to mid-shoulders
                cv2.line(img=frame, pt1=shoulders[0], pt2=mid_shoulders_pt,
                         color=(255, 0, 128), thickness=3)
            if shoulders[1] is not None and biceps[1] is not None:
                # right shoulder joint to biceps
                cv2.line(img=frame, pt1=shoulders[1], pt2=biceps[1],
                         color=(255, 0, 128), thickness=3)
                # right shoulder to mid-shoulders
                cv2.line(img=frame, pt1=shoulders[1], pt2=mid_shoulders_pt,
                         color=(255, 0, 128), thickness=3)
            if biceps[0] is not None and forearms[0] is not None:
                # right biceps joint to forearm
                cv2.line(img=frame, pt1=biceps[0], pt2=forearms[0], color=(255, 0, 128), thickness=3)
            if biceps[1] is not None and forearms[1] is not None:
                # left biceps joint to forearm
                cv2.line(img=frame, pt1=biceps[1], pt2=forearms[1], color=(255, 0, 128), thickness=3)
        except (AttributeError, cv2.error):
            print('Error 001 occurred')
            pass
        # track time
        end = time.time(); exectime = end - start
        print('Execution time for initial detect: ' + str(exectime) + ' secs')
        return frame


if __name__ == '__main__':
    tracked = limb_tracker(path=pic5)
    cv2.imshow(winname='Result', mat=tracked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# FIXME: If shoulder approximation point falls on black pixel in threshold, test surrounding pixels
