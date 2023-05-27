import cv2
import numpy as np


def add_caption_to_img(img, info, name=None, flip_rgb=False):
    """ Adds caption to an image. info is dict with keys and text/array.
        :arg name: if given this will be printed as heading in the first line
        :arg flip_rgb: set to True for inputs with BGR color channels
    """
    offset = 12

    frame = img * 255.0 if img.max() <= 1.0 else img
    if flip_rgb:
        frame = frame[:, :, ::-1]

    # make frame larger if needed
    if frame.shape[0] < 300:
        frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_CUBIC)

    fheight, fwidth = frame.shape[:2]
    frame = np.concatenate([frame, np.zeros((offset * (len(info.keys()) + 2), fwidth, 3))], 0)

    font_size = 0.4
    thickness = 1
    x, y = 5, fheight + 10
    if name is not None:
        cv2.putText(frame, '[{}]'.format(name),
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (100, 100, 0), thickness, cv2.LINE_AA)
    for i, k in enumerate(info.keys()):
        v = info[k]
        key_text = '{}: '.format(k)
        (key_width, _), _ = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX,
                                            font_size, thickness)

        cv2.putText(frame, key_text,
                    (x, y + offset * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (66, 133, 244), thickness, cv2.LINE_AA)

        cv2.putText(frame, str(v),
                    (x + key_width, y + offset * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (100, 100, 100), thickness, cv2.LINE_AA)

    if flip_rgb:
        frame = frame[:, :, ::-1]

    return frame

def add_captions_to_seq(img_seq, info_seq, **kwargs):
    """Adds caption to sequence of image. info_seq is list of dicts with keys and text/array."""
    return [add_caption_to_img(img, info, name='Timestep {:03d}'.format(i), **kwargs) for i, (img, info) in enumerate(zip(img_seq, info_seq))]
