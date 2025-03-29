import argparse

import cv2
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd


def image_match(image0, image1):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")  # 'mps', 'cpu', 'cuda'

    # SuperPoint+LightGlue
    extractor = SuperPoint(max_num_keypoints=None).eval().to(device)  # load the extractor
    matcher = (
        LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1).eval().to(device)
    )  # load the matcher

    # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
    # extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
    # matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    image0 = numpy_image_to_torch(image0[..., ::-1]).to(device)
    image1 = numpy_image_to_torch(image1[..., ::-1]).to(device)

    # extract local features
    feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(image1)

    # match the features
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension
    matches = matches01["matches"]  # indices with shape (K,2)
    points0 = feats0["keypoints"][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1["keypoints"][matches[..., 1]]  # coordinates in image #1, shape (K,2)

    return points0.cpu().numpy(), points1.cpu().numpy()


# feats0
# key 'keypoints' is the keypoints in the first image
# key 'keypoint_scores' is the scores of the keypoints in the first image
# key 'descriptors' is the descriptors of the keypoints in the first image using 256-dimensions
# key 'image_size' is the size of the first image

# matches01
# key 'matches0' is the indices of the matched keypoints index in the second image
# key 'matches1' is the indices of the matched keypoints index in the first image
# key 'matches' is the indices of the matched keypoints index in the first image and the second image
# key 'matching_scores0' is the matching score of the matched keypoints in the second image
# key 'matching_scores1' is the matching score of the matched keypoints in the first image
# key 'scores' is the matching score of the matched keypoints in the first image and the second image
# key 'stop' is the stopping layer of the matcher


def view_points_increment(image0, image1, points0, points1, increment=15, delay=0):
    combined_image = cv2.vconcat([image0, image1])

    if isinstance(points0, torch.Tensor):
        points0 = points0.cpu().numpy()
    if isinstance(points1, torch.Tensor):
        points1 = points1.cpu().numpy()

    if increment == 0:
        increment = len(points0)

    for i in range(0, len(points0), increment):
        show_image = combined_image.copy()
        for p0, p1 in zip(points0[i : i + increment], points1[i : i + increment]):
            p0 = tuple(p0.astype(int))
            p1 = tuple(p1.astype(int))
            cv2.circle(show_image, p0, 10, (0, 0, 255), 2)
            cv2.circle(show_image, (p1[0], p1[1] + image0.shape[0]), 10, (0, 0, 255), 2)
            cv2.line(show_image, p0, (p1[0], p1[1] + image0.shape[0]), (0, 255, 0), 2)
        cv2.imshow("image", show_image)
        cv2.waitKey(delay=delay)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGlue")
    parser.add_argument("--image0", type=str, help="path to image #0")
    parser.add_argument("--image1", type=str, help="path to image #1")
    parser.add_argument("--video", type=str, help="path to video")
    parser.add_argument("--frame0", type=int, help="frame #0")
    parser.add_argument("--frame1", type=int, help="frame #1")
    args = parser.parse_args()

    if args.video:
        if args.frame0 is None or args.frame1 is None:
            raise ValueError("frame0 and frame1 must be provided")
        cap = cv2.VideoCapture()
        cap.open(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame0)
        ret, image0 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame1)
        ret, image1 = cap.read()
    else:
        if not args.image0 or not args.image1:
            raise ValueError("image0 and image1 must be provided")
        image0 = cv2.imread(args.image0)
        image1 = cv2.imread(args.image1)

    points0, points1 = image_match(image0, image1)

    print(len(points0))
    print(len(points1))

    view_points_increment(image0, image1, points0, points1, increment=15, delay=1000)
