import cv2
import os.path
import time
import numpy as np
from sort.sort_tracker import SortTracker
from sort.utils import get_random_colors

SEQUENCES = ['ADL-Rundle-8', 'KITTI-13', 'KITTI-17', 'ETH-Bahnhof', 'ETH-Sunnyday', 'ETH-Pedcross2']
DATA_PATH = "C:/Users/motke/Downloads/2DMOT2015"
DISPLAY = True
COLOURS = get_random_colors(32, 3)
phase = 'train'
total_time = 0.0
total_frames = 0

if __name__ == '__main__':
    # all train
    for seq in SEQUENCES:
        mot_tracker = SortTracker()  # create instance of the SORT tracker
        seq_dets = np.loadtxt('data/%s/det.txt' % (seq), delimiter=',')  # load detections
        print("Processing %s." % (seq))
        for frame in range(int(seq_dets[:, 0].max())):
            frame += 1  # detection and frame numbers begin at 1
            dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
            dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            total_frames += 1

            if (DISPLAY):
                fn = "%s/%s/%s/img1/%06d.jpg" % (DATA_PATH, phase, seq, frame)
                image = cv2.imread(fn)

            start_time = time.time()
            trackers = mot_tracker.update(dets)
            cycle_time = time.time() - start_time
            total_time += cycle_time

            for d in trackers:
                d = d.astype(np.int32)
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]))
                if DISPLAY:

                    cv2.rectangle(image, (d[0], d[1]), (d[2], d[3]), COLOURS[d[4] % 32, :] * 255, 2)
                    cv2.putText(image, "id:" + str(d[4]), (d[0], d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                                lineType=cv2.LINE_AA)
                    cv2.imshow("sort tracker", image)
            key = cv2.waitKey(1)
            if key == 27:
                break
                cv2.destroyAllWindows()

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))
    if (DISPLAY):
        print("Note: to get real runtime results run without the option: --display")
