import base64
import contextlib
from io import BytesIO

import streamlit as st
from PIL import Image

from torchmetrics.functional import f1_score

import torch
import numpy as np
import pandas as pd
from roweeder.data.utils import DataDict

from roweeder.detector import (
    HoughCropRowDetector,
    HoughDetectorDict,
    get_vegetation_detector as get_vegetation_detector_fn,
)
from roweeder.data import get_dataset
from roweeder.data.phenobench import PhenoBenchDataset
from roweeder.labeling import get_drawn_img, label_from_row, label, save_and_label
from roweeder.visualize import map_grayscale_to_rgb


def change_state(src, dest):
    st.session_state[dest] = st.session_state[src]
    st.session_state["i"] = st.session_state[src]


@st.cache_resource
def get_vegetation_detector(ndvi_threshold=0.5, exg_threshold=0.3, exg_min_area=100):
    name = st.session_state["vegetation_detector"]
    
    # Ensure NDVIDetector is not used with PhenoBench
    if st.session_state.get("modality") == "PhenoBench" and name != "ExGDetector":
        st.warning("NDVI is not supported for PhenoBench. Switching to ExGDetector.")
        name = "ExGDetector"
        st.session_state["vegetation_detector"] = name

    if name == "NDVIDetector":
        # Default indices for WeedMap (R,G,B,NIR,RE)
        nir_idx = 3
        red_idx = 0
        # If using PhenoBench (RGB only), set indices to valid RGB channels (dummy NDVI)
        if st.session_state.get("modality") == "PhenoBench":
            nir_idx = 0
            red_idx = 2
        params = {
            "threshold": ndvi_threshold,
            "nir_idx": nir_idx,
            "red_idx": red_idx,
        }
    elif name == "ExGDetector":
        params = {
            "threshold": exg_threshold,
            "min_area": exg_min_area,
        }
    else:
        raise ValueError(f"Unknown vegetation detector: {name}")
    return get_vegetation_detector_fn(name, params)


def get_thumbnail(path):
    i = Image.open(path)
    i.thumbnail((150, 150), Image.LANCZOS)
    return i


def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, "jpeg")
        return base64.b64encode(buffer.getvalue()).decode()


def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'


def gt_fix(gt):
    gt[gt == 10000] = 1
    gt[gt == 240] = 2
    return gt


def display_prediction():
    st_state = st.session_state
    if st_state["img_name"] != "":
        found = False
        for k, name in enumerate(st_state["images"]):
            if st_state["img_name"] in name:
                i = k
                found = True
                st_state["i"] = i
        if not found:
            st.write("img not found")
            return
    else:
        try:
            i = st_state["i"]
        except KeyError:
            i = 0
            st_state["i"] = i

    col1, col2, col3, col4, col5 = st.columns(5)
    data_dict = st_state["dataset"][i]
    img = data_dict.image
    gt = data_dict.target

    mask = st.session_state["labeller"](img)
    st.write(mask.shape)
    detector = HoughCropRowDetector(
        threshold=st.session_state["threshold"],
        crop_detector=st.session_state["labeller"],
        step_theta=st.session_state["step_theta"],
        step_rho=st.session_state["step_rho"],
        angle_error=st.session_state["angle_error"],
        clustering_tol=st.session_state["clustering_tol"],
        uniform_significance=st.session_state["uniform_significance"],
        theta_reduction_threshold=st.session_state["theta_reduction_threshold"],
        theta_value=st.session_state["theta_value"],
    )
    res = detector.predict_from_mask(mask)
    lines = res[HoughDetectorDict.LINES]
    original_lines = res[HoughDetectorDict.ORIGINAL_LINES]
    uniform_significance = res[HoughDetectorDict.UNIFORM_SIGNIFICANCE]
    zero_reason = res[HoughDetectorDict.ZERO_REASON]

    gt = gt_fix(torch.tensor(np.array(gt))).cuda()

    to_draw_gt = (gt.cpu().numpy()).astype(np.uint8)
    to_draw_gt = map_grayscale_to_rgb(to_draw_gt)
    to_draw_gt = np.squeeze(to_draw_gt)  # Remove extra dimensions
    st.image(Image.fromarray(to_draw_gt), width=300)
    to_draw_mask = mask.cpu().numpy().astype(np.uint8)
    line_mask = get_drawn_img(
        torch.zeros_like(torch.tensor(to_draw_mask)).numpy(), lines, color=(255, 0, 255)
    )
    argmask = mask[0].type(torch.uint8)
    weed_map, weed_map_slic, slic = label_from_row(img, argmask, torch.tensor(line_mask).permute(2, 0, 1)[0])
    pred = weed_map.argmax(dim=0)
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    if gt.ndim == 2:
        gt = gt.unsqueeze(0)
    f1 = f1_score(
        pred.cuda(),
        gt.cuda(),
        num_classes=3,
        average="macro",
        task="multiclass",
        multidim_average="global",
    )
    weed_map = weed_map.argmax(dim=0).cpu().numpy().astype(np.uint8)
    weed_map = map_grayscale_to_rgb(
        weed_map, mapping={1: (0, 255, 0), 2: (255, 0, 0)}
    ).transpose(2, 0, 1)
    weed_map_lines = get_drawn_img(weed_map, lines, color=(255, 0, 255))
    to_draw_mask = to_draw_mask[0]
    weed_map = np.moveaxis(weed_map, 0, -1)

    st.write(data_dict.name)
    st.write("f1 score: ", f1)
    st.write("uniform_significance: ", uniform_significance)
    st.write("zero_reason: ", zero_reason)
    with col1:
        st.write("## Image")
        output_img = (img[:3].squeeze(0).permute(1, 2, 0).numpy() * 255).astype(
            np.uint8
        )
        output_img = Image.fromarray(output_img)
        st.image(output_img, width=300)
    with col2:
        st.write("## GT")
        st.image(Image.fromarray(to_draw_gt), width=300)
    with col3:
        st.write("## Mask")
        st.image(Image.fromarray(to_draw_mask), width=300)
    with col4:
        st.write("## Prediction")
        st.image(weed_map, width=300)
    with col5:
        st.write("## Lines")
        st.image(Image.fromarray(weed_map_lines), width=300)
        
    st.write("## Lines")
    st.dataframe(pd.DataFrame(lines.cpu(), columns=["rho", "theta"]))
    st.write("## Original Lines")
    st.dataframe(
        pd.DataFrame(
            (original_lines.cpu() if original_lines is not None else []),
            columns=["rho", "theta"],
        )
    )


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    
    # Add PhenoBench to the options
    default_roots = {
        "PhenoBench": "dataset/PhenoBench",
        "New Dataset": "dataset/patches/512"
    }

    with st.sidebar:
        # Add PhenoBench to the selectbox
        st.selectbox("Modality", ["PhenoBench", "New Dataset"], key="modality")
        
        st.text_input(
            value=default_roots[st.session_state["modality"]],
            key="root",
            label="root",
        )

        # --- NEW LOGIC TO CHOOSE THE CORRECT DATASET ---
        if st.session_state["modality"] == "PhenoBench":
            # For PhenoBench, we use a simple train/val split selector
            split = st.selectbox("Split", ["train", "val"], key="phenobench_split")
            dataset = PhenoBenchDataset(root_dir=st.session_state["root"], split=split)
        else:
            # Original logic for the WeedMap-style dataset
            fields = st.multiselect(
                "Fields",
                ["000", "001", "002", "003", "004"],
                key="fields",
                default=["000", "001", "002", "003", "004"],
            )
            dataset = get_dataset(
                st.session_state["root"], st.session_state["modality"], fields
            )
        
        st.session_state["dataset"] = dataset

        st.slider(
            "i",
            max_value=len(st.session_state["dataset"])-1,
            step=1,
            key="slider_i",
            on_change=lambda: change_state("slider_i", "number_i"),
        )  # 👈 this is a widget
        st.number_input(
            "i",
            key="number_i",
            value=0,
            on_change=lambda: change_state("number_i", "slider_i"),
        )
        st.text_input(value="", label="img_name", key="img_name")
        if st.session_state["modality"] == "PhenoBench":
            if st.session_state.get("vegetation_detector") != "ExGDetector":
                st.warning("NDVI is not supported for PhenoBench. Switching to ExGDetector.")
                st.session_state["vegetation_detector"] = "ExGDetector"
            st.selectbox(
                "Vegetation Detector",
                ["ExGDetector"],
                key="vegetation_detector",
                index=0,
                disabled=True
            )
        else:
            st.selectbox(
                "Vegetation Detector",
                ["NDVIDetector", "ExGDetector"],
                key="vegetation_detector",
                index=0,
            )
        ndvi_threshold = st.slider(
            "ndvi_threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=0.6,
            key="ndvi_threshold",
        )
        exg_threshold = st.slider(
            "exg_threshold",
            min_value=-1.0,
            max_value=2.0,
            step=0.01,
            value=-2.0,
            key="exg_threshold",
        )
        exg_min_area = st.number_input(
            "exg_min_area",
            min_value=0,
            max_value=1000,
            value=60,
            step=1,
            key="exg_min_area",
        )
        st.session_state["labeller"] = get_vegetation_detector(
            ndvi_threshold=ndvi_threshold,
            exg_threshold=exg_threshold,
            exg_min_area=exg_min_area,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.slider(
            "threshold", min_value=0, max_value=255, step=1, value=150, key="threshold"
        )
        st.slider(
            "step_theta", min_value=1, max_value=10, step=1, value=1, key="step_theta"
        )
        st.slider(
            "step_rho", min_value=1, max_value=10, step=1, value=1, key="step_rho"
        )
        # New theta override option: Checkbox to enable override
        override_theta = st.checkbox("Override theta calculation?", key="override_theta")
        if override_theta:
            # The number_input widget creates st.session_state["theta_value"] automatically.
            theta_value = st.number_input("Fixed theta value", key="theta_value", value=0.0, step=0.01)
        else:
            theta_value = None
    with col2:
        st.slider(
            "angle_error", min_value=0, max_value=10, step=1, value=3, key="angle_error"
        )
        st.slider(
            "clustering_tol",
            min_value=0,
            max_value=50,
            step=1,
            value=3,
            key="clustering_tol",
        )
        st.slider(
            "uniform_significance",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=0.1,
            key="uniform_significance",
        )
        st.slider(
            "theta_reduction_threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=1.0,
            key="theta_reduction_threshold",
        )

    display_prediction()

    st.text_input(value="dataset/generated", label="out_dir", key="out_dir")
    if st.button("label"):
        bar = st.progress(0)
        for i in save_and_label(
            outdir=st.session_state["out_dir"],
            plant_detector_params=dict(
                name=st.session_state["vegetation_detector"],
                params=dict(
                    checkpoint=st.session_state["checkpoint"],
                    threshold=st.session_state["ndvi_threshold"],
                ),
            ),
            hough_detector_params=dict(
                threshold=st.session_state["threshold"],
                step_theta=st.session_state["step_theta"],
                step_rho=st.session_state["step_rho"],
                angle_error=st.session_state["angle_error"],
                clustering_tol=st.session_state["clustering_tol"],
                uniform_significance=st.session_state["uniform_significance"],
                theta_reduction_threshold=st.session_state["theta_reduction_threshold"],
                theta_value=st.session_state["theta_value"],
            ),
            dataset_params=dict(
                root=st.session_state["root"],
                modality=st.session_state["modality"],
                fields=st.session_state["fields"],
            ),
            interactive=True,
        ):
            bar.progress(i / len(st.session_state["dataset"]))

    # st.slider('theta', min_value=0.0, max_value=5.0, step=0.01, value=0.0, key="theta")
    # st.slider('rho', min_value=0, max_value=1000, step=1, value=0, key="rho")
    # blank = np.zeros((3, 300, 300), np.uint8)
    # blank = get_drawn_img(blank, [(st.session_state['rho'], st.session_state['theta'])])
    # st.image(blank, width=300)
    # st.write(np.rad2deg(st.session_state['theta']))
    # st.write(np.sin(2*st.session_state['theta']))
    # st.write((1 - np.abs(np.sin(2*st.session_state['theta']))))
    # st.slider('min_reduce', min_value=0.0, max_value=1.0, step=0.01, value=0.5, key="min_reduce")
    # min_reduce = st.session_state['min_reduce']
    # st.write((1 - np.abs(np.sin(2*st.session_state['min_reduce']))) * (1 - min_reduce) / 1 + min_reduce)
