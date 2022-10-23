import sys
import time
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage import measure, draw
from skimage.measure import regionprops
from skimage.filters import threshold_minimum
from skimage.measure import regionprops
from skimage.exposure import equalize_adapthist
from skimage.color import label2rgb
from scipy import optimize
from scipy.ndimage import binary_fill_holes
import torch
from torchvision.transforms import Resize
from torchvision.transforms import functional as F
from eye.UI.models.get_model import get_arch
from eye.UI.utils.model_saving_loading import load_model
from eye.UI.utils import paired_transforms_tv04 as p_tr


def get_circ(binary):

    image = binary.astype(int)
    regions = measure.regionprops(image)
    bubble = regions[0]

    y0, x0 = bubble.centroid
    r = bubble.major_axis_length / 2.0

    def cost(params):
        x0, y0, r = params
        coords = draw.circle(y0, x0, r, shape=image.shape)
        template = np.zeros_like(image)
        template[coords] = 1
        return -np.sum(template == image)

    x0, y0, r = optimize.fmin(cost, (x0, y0, r))
    return x0, y0, r


def create_circular_mask(sh, center=None, radius=None):

    h, w = sh
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def get_fov(img):
    im_s = img.size
    if max(im_s) > 500:
        img = Resize(500)(img)

    with np.errstate(divide="ignore"):
        im_v = equalize_adapthist(np.array(img))[:, :, 1]
        # im_v = equalize_adapthist(rgb2hsv(np.array(img))[:, :, 2])
    thresh = threshold_minimum(im_v)
    binary = binary_fill_holes(im_v > thresh)

    x0, y0, r = get_circ(binary)
    fov = create_circular_mask(binary.shape, center=(x0, y0), radius=r)

    return Resize(im_s[::-1])(Image.fromarray(fov))


def crop_to_fov(img, mask):
    mask = np.array(mask).astype(int)
    minr, minc, maxr, maxc = regionprops(mask)[0].bbox
    im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
    return im_crop, [minr, minc, maxr, maxc]


def flip_ud(tens):
    return torch.flip(tens, dims=[1])


def flip_lr(tens):
    return torch.flip(tens, dims=[2])


def flip_lrud(tens):
    return torch.flip(tens, dims=[1, 2])


def create_pred(model, tens, mask, coords_crop, original_sz, bin_thresh, tta="no"):
    act = torch.sigmoid if model.n_classes == 1 else torch.nn.Softmax(dim=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        logits = model(tens.unsqueeze(dim=0).to(device)).squeeze(dim=0)
    pred = act(logits)

    if tta != "no":
        with torch.no_grad():
            logits_lr = (
                model(tens.flip(-1).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-1)
            )
            logits_ud = (
                model(tens.flip(-2).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-2)
            )
            logits_lrud = (
                model(tens.flip(-1).flip(-2).unsqueeze(dim=0).to(device))
                .squeeze(dim=0)
                .flip(-1)
                .flip(-2)
            )

        if tta == "from_logits":
            mean_logits = torch.mean(
                torch.stack([logits, logits_lr, logits_ud, logits_lrud]), dim=0
            )
            pred = act(mean_logits)
        elif tta == "from_preds":
            pred_lr = act(logits_lr)
            pred_ud = act(logits_ud)
            pred_lrud = act(logits_lrud)
            pred = torch.mean(torch.stack([pred, pred_lr, pred_ud, pred_lrud]), dim=0)
        else:
            raise NotImplementedError
    pred = (
        pred.detach().cpu().numpy()[-1]
    )  # this takes last channel in multi-class, ok for 2-class
    # Orders: 0: NN, 1: Bilinear(default), 2: Biquadratic, 3: Bicubic, 4: Biquartic, 5: Biquintic
    pred = resize(pred, output_shape=original_sz, order=3)
    full_pred = np.zeros_like(mask, dtype=float)
    full_pred[coords_crop[0] : coords_crop[2], coords_crop[1] : coords_crop[3]] = pred
    full_pred[~mask.astype(bool)] = 0
    full_pred_bin = full_pred > bin_thresh
    return full_pred, full_pred_bin


def create_pred_av(model, tens, mask, coords_crop, original_sz, tta="no"):
    act = torch.nn.Softmax(dim=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        logits = model(tens.unsqueeze(dim=0).to(device)).squeeze(dim=0)
    prob = act(logits)

    if tta != "no":
        with torch.no_grad():
            logits_lr = (
                model(tens.flip(-1).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-1)
            )
            logits_ud = (
                model(tens.flip(-2).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-2)
            )
            logits_lrud = (
                model(tens.flip(-1).flip(-2).unsqueeze(dim=0).to(device))
                .squeeze(dim=0)
                .flip(-1)
                .flip(-2)
            )
        if tta == "from_logits":
            mean_logits = torch.mean(
                torch.stack([logits, logits_lr, logits_ud, logits_lrud]), dim=0
            )
            prob = act(mean_logits)
        elif tta == "from_probs":
            prob_lr = act(logits_lr)
            prob_ud = act(logits_ud)
            prob_lrud = act(logits_lrud)
            prob = torch.mean(torch.stack([prob, prob_lr, prob_ud, prob_lrud]), dim=0)
        else:
            raise NotImplementedError
    # prob is now n_classes x h_train x w_train
    prob = prob.detach().cpu().numpy()
    # Orders: 0: NN, 1: Bilinear(default), 2: Biquadratic, 3: Bicubic, 4: Biquartic, 5: Biquintic

    prob_0 = resize(prob[0], output_shape=original_sz, order=3)
    prob_1 = resize(prob[1], output_shape=original_sz, order=3)
    prob_2 = resize(prob[2], output_shape=original_sz, order=3)
    prob_3 = resize(prob[3], output_shape=original_sz, order=3)

    full_prob_0 = np.zeros_like(mask, dtype=float)
    full_prob_1 = np.zeros_like(mask, dtype=float)
    full_prob_2 = np.zeros_like(mask, dtype=float)
    full_prob_3 = np.zeros_like(mask, dtype=float)

    full_prob_0[
        coords_crop[0] : coords_crop[2], coords_crop[1] : coords_crop[3]
    ] = prob_0
    full_prob_0[~mask.astype(bool)] = 0
    full_prob_1[
        coords_crop[0] : coords_crop[2], coords_crop[1] : coords_crop[3]
    ] = prob_1
    full_prob_1[~mask.astype(bool)] = 0
    full_prob_2[
        coords_crop[0] : coords_crop[2], coords_crop[1] : coords_crop[3]
    ] = prob_2
    full_prob_2[~mask.astype(bool)] = 0
    full_prob_3[
        coords_crop[0] : coords_crop[2], coords_crop[1] : coords_crop[3]
    ] = prob_3
    full_prob_3[~mask.astype(bool)] = 0

    # full_prob_1 corresponds to uncertain pixels, we redistribute probability between prob_1 and prob_2
    full_prob_2 += 0.5 * full_prob_1
    full_prob_3 += 0.5 * full_prob_1
    full_prob = np.stack(
        [full_prob_0, full_prob_2, full_prob_3], axis=2
    )  # background, artery, vein

    full_pred = np.argmax(full_prob, axis=2)
    full_rgb_pred = label2rgb(full_pred, colors=["black", "red", "blue"])

    return np.clip(full_prob, 0, 1), full_rgb_pred


def segment(img):
    """segment function used for segmenting vessel of eye

    segmentation without classifying of vessel eye

    Parameters
    ----------
    img : PIL image
        the image that read with PIL

    Returns
    -------
    torch.tensor
        the segmented image
    """
    device = torch.device("cpu")
    bin_thresh = 0.4196
    tta = "from_preds"
    model_name = "wnet"
    im_size = (512, 512)

    if isinstance(im_size, tuple) and len(im_size) == 1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size) == 2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit("im_size should be a number or a tuple of two numbers")

    mask = get_fov(img)
    mask = np.array(mask).astype(bool)

    img, coords_crop = crop_to_fov(img, mask)
    original_sz = img.size[1], img.size[0]

    rsz = p_tr.Resize(tg_size)
    img = rsz(img)
    im_tens = F.to_tensor(img)

    model = get_arch(model_name).to(device)
    if model_name == "wnet":
        model.mode = "eval"

    model, stats = load_model(
        model,
        "eye/UI/experiments/wnet_drive/",
        device,
    )

    model.eval()

    start_time = time.perf_counter()
    full_pred, full_pred_bin = create_pred(
        model, im_tens, mask, coords_crop, original_sz, bin_thresh=bin_thresh, tta=tta
    )

    return full_pred


def segment_av(img):
    """segment_av function used for segmenting vessel of eye


    segmentation with classifying of vessel eye

    Parameters
    ----------
    img : PIL image
        the image that read with PIL

    Returns
    -------
    torch.tensor
        the segmented image
    """
    device = torch.device("cpu")

    tta = "from_probs"
    model_name = "big_wnet"
    im_size = (512, 512)

    if isinstance(im_size, tuple) and len(im_size) == 1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size) == 2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit("im_size should be a number or a tuple of two numbers")

    mask = get_fov(img)
    mask = np.array(mask).astype(bool)

    img, coords_crop = crop_to_fov(img, mask)
    original_sz = img.size[1], img.size[0]

    rsz = p_tr.Resize(tg_size)
    img = rsz(img)

    im_tens = F.to_tensor(img)

    model = get_arch(model_name, n_classes=4).to(device)
    if model_name == "big_wnet":
        model.mode = "eval"

    model, stats = load_model(model, "eye/UI/experiments/big_wnet_hrf_av_1024/", device)
    model.eval()

    start_time = time.perf_counter()
    full_pred, full_pred_bin = create_pred_av(
        model, im_tens, mask, coords_crop, original_sz, tta=tta
    )

    return full_pred
