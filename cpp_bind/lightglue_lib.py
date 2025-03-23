import numpy as np
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# SuperPoint+LightGlue
extractor = SuperPoint(device, max_num_keypoints=None).eval().to(device)  # load the extractor

matcher = (
    LightGlue(device, features="superpoint", depth_confidence=-1, width_confidence=-1)
    .eval()
    .to(device)
)  # load the matcher


def feature_extract(image):
    image = numpy_image_to_torch(np.array(image)[..., ::-1]).to(device)
    feats = extractor.extract(image)
    return feats["keypoints"].cpu().numpy()[0], feats["descriptors"].cpu().numpy()[0]


def image_match(image0, image1, keypoints0, keypoints1, descriptors0, descriptors1):
    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]

    image0 = numpy_image_to_torch(np.array(image0)[..., ::-1]).to(device)
    image1 = numpy_image_to_torch(np.array(image1)[..., ::-1]).to(device)

    feats0 = {
        "keypoints": torch.from_numpy(np.array([keypoints0]).astype(np.float32)).to(device),
        "descriptors": torch.from_numpy(np.array([descriptors0]).astype(np.float32)).to(device),
        "image_size": torch.Tensor([[image0.shape[2], image0.shape[1]]]).to(device),
    }
    feats1 = {
        "keypoints": torch.from_numpy(np.array([keypoints1]).astype(np.float32)).to(device),
        "descriptors": torch.from_numpy(np.array([descriptors1]).astype(np.float32)).to(device),
        "image_size": torch.Tensor([[image1.shape[2], image1.shape[1]]]).to(device),
    }

    # match the features
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension
    matches = matches01["matches"]  # indices with shape (K,2)
    scores = matches01["scores"]
    # points0 = feats0["keypoints"][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    # points1 = feats1["keypoints"][matches[..., 1]]  # coordinates in image #1, shape (K,2)

    return matches.cpu().numpy(), scores.cpu().numpy()


if __name__ == "__main__":
    import numpy as np

    image0 = np.load("frame_0_12.jpg.npz")["arr_0"]

    keypoints, descriptors = feature_extract(image0)

    print(keypoints)
    print(descriptors)

    matches = image_match(image0, image0, keypoints, keypoints, descriptors, descriptors)

    print(matches)
