import io
import itertools
import datetime
from dateutil.parser import parse
import subprocess
import time
import urllib.request
from threading import Thread
from typing import Tuple
from os import path
import os
import cv2
import grpc
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw

import inference_pb2_grpc
import management_pb2
import management_pb2_grpc
import streamlit as st

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)

    def getframe(self):
        return self.frame


st.set_page_config(layout="wide")

TEST_IMAGES = {
    "Front snap": "http://192.168.10.184/snap.jpeg",
    "Dock snap": "http://192.168.10.198/snap.jpeg",
    "Driveway snap": "http://192.168.10.246/snap.jpeg",
    "Backyard stock": "https://i.pinimg.com/564x/44/2c/24/442c24d9456bf2635bceb0cedfdee09e.jpg",
    "Labeled": "",
}

directory = f'/mnt/nas_downloads/data/unifitest'
i = 1
for filename in os.listdir(directory):
    if (filename.endswith(".jpg") or filename.endswith(".png")) and "_object_" in filename:
        TEST_IMAGES[f"Crop #{i}"] = f"{directory}/{filename}"
        i = i + 1
    else:
        continue

CONTAINER = "serve_xaser"

RED = (255, 0, 0)  # For objects within the ROI
GREEN = (0, 255, 0)  # For ROI box
YELLOW = (255, 255, 0)  # For objects outside the ROI

DEFAULT_ROI_Y_MIN = 0.0
DEFAULT_ROI_Y_MAX = 1.0
DEFAULT_ROI_X_MIN = 0.0
DEFAULT_ROI_X_MAX = 1.0
DEFAULT_ROI = (
    DEFAULT_ROI_Y_MIN,
    DEFAULT_ROI_X_MIN,
    DEFAULT_ROI_Y_MAX,
    DEFAULT_ROI_X_MAX,
)


def paginator(label, items, items_per_page=10, on_sidebar=True):
    # Figure out where to display the paginator
    if on_sidebar:
        location = st.sidebar.empty()
    else:
        location = st.empty()

    # Display a pagination selectbox in the specified location.
    items = list(items)
    n_pages = len(items)
    n_pages = (len(items) - 1) // items_per_page + 1
    def page_format_func(i): return "Page %s" % i
    page_number = location.selectbox(
        label, range(n_pages), format_func=page_format_func)

    # Iterate over the items in the page to let the user display them.
    min_index = page_number * items_per_page
    max_index = min_index + items_per_page
    return itertools.islice(enumerate(items), min_index, max_index)


def get_inference_stub():
    channel = grpc.insecure_channel('localhost:7070', options=[
        ('grpc.max_send_message_length', int(2147483647)),
        ('grpc.max_receive_message_length', int(2147483647)),
    ])
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub


def get_management_stub():
    channel = grpc.insecure_channel('localhost:7071')
    stub = management_pb2_grpc.ManagementAPIsServiceStub(channel)
    return stub


def infer(stub, model_name, model_input):
    url = 'http://localhost:8080/predictions/{}'.format(model_name)
    prediction = requests.post(url, data=model_input).text
    return prediction


def register(stub, model_name, local=False):
    if local:
        url = "{}".format(model_name)
    else:
        url = "https://torchserve.s3.amazonaws.com/mar_files/{}.mar".format(
            model_name)
    params = {
        'url': url,
        'initial_workers': 1,
        'synchronous': True,
        'model_name': model_name.replace(".mar", "")
    }
    response = stub.RegisterModel(
        management_pb2.RegisterModelRequest(**params))
    return response


def unregister(stub, model_name):
    response = stub.UnregisterModel(
        management_pb2.UnregisterModelRequest(model_name=model_name))
    return response


def get_objects(predictions: list, img_width: int, img_height: int):
    """Return objects with formatting and extra info."""
    objects = []
    decimal_places = 3
    for pred in predictions:
        if isinstance(pred, str):  # this is image class not object detection so no objects
            return objects
        name = list(pred.keys())[0]

        box_width = pred[name][2]-pred[name][0]
        box_height = pred[name][3]-pred[name][1]
        box = {
            "height": round(box_height / img_height, decimal_places),
            "width": round(box_width / img_width, decimal_places),
            "y_min": round(pred[name][1] / img_height, decimal_places),
            "x_min": round(pred[name][0] / img_width, decimal_places),
            "y_max": round(pred[name][3] / img_height, decimal_places),
            "x_max": round(pred[name][2] / img_width, decimal_places),
        }
        box_area = round(box["height"] * box["width"], decimal_places)
        centroid = {
            "x": round(box["x_min"] + (box["width"] / 2), decimal_places),
            "y": round(box["y_min"] + (box["height"] / 2), decimal_places),
        }
        confidence = round(pred['score'], decimal_places)

        objects.append(
            {
                "bounding_box": box,
                "box_area": box_area,
                "centroid": centroid,
                "name": name,
                "confidence": confidence,
            }
        )
    return objects


def draw_box(
    draw: ImageDraw,
    box: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
    text: str = "",
    color: Tuple[int, int, int] = (255, 255, 0),
) -> None:
    """
    Draw a bounding box on and image.
    The bounding box is defined by the tuple (y_min, x_min, y_max, x_max)
    where the coordinates are floats in the range [0.0, 1.0] and
    relative to the width and height of the image.
    For example, if an image is 100 x 200 pixels (height x width) and the bounding
    box is `(0.1, 0.2, 0.5, 0.9)`, the upper-left and bottom-right coordinates of
    the bounding box will be `(40, 10)` to `(180, 50)` (in (x,y) coordinates).
    """

    line_width = 3
    font_height = 8
    y_min, x_min, y_max, x_max = box
    (left, right, top, bottom) = (
        x_min * img_width,
        x_max * img_width,
        y_min * img_height,
        y_max * img_height,
    )
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=line_width,
        fill=color,
    )
    if text:
        draw.text(
            (left + line_width, abs(top - line_width - font_height)), text, fill=color
        )

models = ["none", "Remote: resnet-18", "Remote: alexnet", "Remote: densenet161", "Remote: vgg11_v2", "Remote: squeezenet1_1",
          "Remote: resnet-152-batch_v2", "Remote: fastrcnn", "Remote: fcn_resnet_101_scripted", "Remote: maskrcnn"]
if path.exists("/usr/bin/docker"):
    # outside container
    result = subprocess.run(['/usr/bin/docker', 'exec', '-it', CONTAINER, 'ls', '-1',
                             '/home/model-server/model-store'], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
else:
    # inside container
    result = subprocess.run(['ls', '-1',
                             '/home/model-server/model-store'], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')

result = [f"Local: {file.strip()}" for file in result if len(file) > 0]
models.extend(result)

images = TEST_IMAGES
pick_img = st.sidebar.radio("Which image?", [x for x in images.keys()])

img_file_buffer = st.sidebar.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg"])

models = ['none']
tochserve = True
try:
    for model in eval(get_management_stub().ListModels(management_pb2.ListModelsRequest(**params)).msg)['models']:
        models.append(model['modelName'])

    model = st.sidebar.selectbox('Zoo', models)

    if st.sidebar.button('Register'):
        source, model_name = model.split(" ")
        if source == "Local:":
            r = register(get_management_stub(), model_name, True)
        else:
            r = register(get_management_stub(), model_name, False)
        st.write(r)

    params = {
        'limit': 100,
        'next_page_token': 0
    }

except:
    st.sidebar.write("Torchserve is down")
    tochserve = False
    pass

model = st.sidebar.selectbox('Model to use', models)

if (tochserve):
    if st.sidebar.button('De-register'):
        r = unregister(get_management_stub(), model)
        st.write(r)

if pick_img:
    item = images[pick_img]
    if isinstance(item, str):
        filename = item
        stream = None
    else:
        filename = item[0]
        stream = item[1]



# Get ROI info
st.sidebar.title("ROI")
ROI_X_MIN = st.sidebar.slider("x_min", 0.0, 1.0, DEFAULT_ROI_X_MIN)
ROI_Y_MIN = st.sidebar.slider("y_min", 0.0, 1.0, DEFAULT_ROI_Y_MIN)
ROI_X_MAX = st.sidebar.slider("x_max", 0.0, 1.0, DEFAULT_ROI_X_MAX)
ROI_Y_MAX = st.sidebar.slider("y_max", 0.0, 1.0, DEFAULT_ROI_Y_MAX)
ROI_TUPLE = (
    ROI_Y_MIN,
    ROI_X_MIN,
    ROI_Y_MAX,
    ROI_X_MAX,
)
ROI_DICT = {
    "x_min": ROI_X_MIN,
    "y_min": ROI_Y_MIN,
    "x_max": ROI_X_MAX,
    "y_max": ROI_Y_MAX,
}

col1, col2 = st.beta_columns(2)

image_placeholder = col1.empty()
frame_placeholder = col1.empty()
timer_placeholder = col2.empty()
data_placeholder = col2.empty()

if stream is not None:
    video = cv2.VideoCapture(stream)
    video.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    fps = video.get(cv2.CAP_PROP_FPS)

tic = time.perf_counter()
frameFpsStart = 0
frameIdProcStart = 0
frameIdProc = 0
frameId = 0

skip = 0

if "Labeled" in pick_img:
    labels = "/mnt/localshared/data/hassio/tstreamer/labels.csv"
    excludes_file="/mnt/localshared/data/hassio/tstreamer/exclude.lst"

    df = pd.read_csv(labels, header=0)
    df.columns = ['stamp', 'uuid', 'puuid', 'entity', 'model', 'confidence', 'similarity', 'label', 'area', 'x1', 'y1', 'x2', 'y2']
    now = datetime.datetime.now() - datetime.timedelta(days=3)
    df = df[df['stamp'] >= now.strftime("%Y-%m-%d")]
    df['gpuuid'] = df['puuid'].map(df.set_index('uuid')['puuid'])
    roundto=30
    df['x1c'] = round(df['x1']/roundto)*roundto
    df['x2c'] = round(df['x2']/roundto)*roundto
    df['y1c'] = round(df['y1']/roundto)*roundto
    df['y2c'] = round(df['y2']/roundto)*roundto
    df = df[(df.model == "yolov5x-1280")]
    dfc = df.pivot_table(index=['entity', 'label', 'x1c', 'y1c', 'x2c', 'y2c'], values=['uuid'], aggfunc='count')
    dfc.sort_values(ascending=False, by=['uuid'], inplace=True)
    dfc=dfc.head(50)
    df = pd.merge(df, dfc, on=['entity', 'label', 'x1c', 'y1c', 'x2c', 'y2c'], how='inner')

    for index, row in dfc.iterrows():
        dfcu = df[df['uuid_y'] == row.uuid]
        dfcu['filename'] = "/mnt/localshared/data/hassio/tstreamer/" + dfcu['entity'] + "_" + dfcu['stamp'] + "_" + dfcu['model'] + "_object_" + dfcu['label'] + "_" + dfcu['uuid_x'] + "_pad.jpg"
        dfcu['pfilename'] = "/mnt/localshared/data/hassio/tstreamer/" + dfcu['entity'] + "_" + dfcu['stamp'] + "_nobox_" + dfcu['puuid'] + ".jpg"
        dfcu['exists'] = dfcu['filename'].map(os.path.isfile)
        dfcu = dfcu[dfcu['exists'] == 1].head(5)

        with open(dfcu.iloc[0]['pfilename'],'rb') as img_file:
            img_file.seek(163)
            a = img_file.read(2)
            height = (a[0] << 8) + a[1]
            a = img_file.read(2)
            width = (a[0] << 8) + a[1]

            cx = ((index[2] + index[4]) / 2) / width
            cy = ((index[3] + index[5]) / 2) / height

            exclduestr = f"(\"{index[0]}\" not in args.name or \"{index[1]}\" not in obj[\"name\"] or not ({cx-roundto/width} <= obj[\"centroid\"][\"x\"] <= {cx+roundto/width} and {cy-roundto/height} <= obj[\"centroid\"][\"y\"] <= {cy+roundto/height})) and"

            with open(excludes_file) as excludesfile:
                if exclduestr in excludesfile.read():
                    continue

            st.write(f"{row['uuid']}")

            st.image(dfcu.filename.values.tolist(), width=100)

            if st.button(f"Exclude {index})"):
                st.write(f"Excluding")
                with open(excludes_file, "a") as excludesfile:
                    excludesfile.write(f"##### automated exclude\n")
                    excludesfile.write(f"{exclduestr}\n")
else:
    while True:
        if stream is not None:
            success, npimage = video.read()
            npimage = cv2.cvtColor(npimage, cv2.COLOR_BGR2RGB)
            frameId = frameId + 1
            fps_real = 1/(time.perf_counter()-tic) * (frameId-frameFpsStart)
            if fps_real < fps and skip < 15:
                skip = skip + 1
                continue
            skip = 0
            frameIdProc = frameIdProc + 1
            fps_actual = 1/(time.perf_counter()-tic) * \
                (frameIdProc-frameIdProcStart)
            frame_placeholder.write(
                f"{fps_real:0.2f} cycle FPS vs {fps_actual:0.2f} actual FPS vs stream {fps} FPS")
            frameFpsStart = frameId
            frameIdProcStart = frameIdProc
            tic = time.perf_counter()
            pil_image = Image.fromarray(npimage)
        else:
            if img_file_buffer is not None:
                pil_image = Image.open(img_file_buffer)
            else:
                if "http" in filename:
                    pil_image = Image.open(urllib.request.urlopen(filename))
                else:
                    pil_image = Image.open(filename)



        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        draw = ImageDraw.Draw(pil_image)

        if model is not None and model != "none":
            infers = time.perf_counter()
            response = eval(infer(get_inference_stub(), model, img_byte_arr))
            inferf = time.perf_counter()
            objects = get_objects(response, pil_image.width, pil_image.height)

            for obj in objects:
                name = obj["name"]
                confidence = obj["confidence"]
                box = obj["bounding_box"]
                box_label = f"{name} {confidence}"
                draw_box(draw, (box["y_min"], box["x_min"], box["y_max"], box["x_max"]),
                        pil_image.width, pil_image.height, text=box_label, color=YELLOW,)

            if response is list or isinstance(response, dict):
                df = pd.DataFrame(response.items())
            elif isinstance(response, str):
                col2.write(response)
            else:
                df = pd.DataFrame()
                for pred in response:
                    label = list(pred.keys())[0]
                    row = [label, f"{pred['score']:0.2f}", f"{pred[label][0]/pil_image.width:0.2f}",
                        f"{pred[label][1]/pil_image.height:0.2f}", f"{pred[label][2]/pil_image.width:0.2f}", f"{pred[label][3]/pil_image.height:0.2f}"]
                    row = pd.DataFrame(row).T
                    df = df.append(row)
                    # col2.write(row)

            timer_placeholder.write(
                f"Infer in {inferf - infers:0.4f}s or {1/(inferf-infers):0.4f} FPS")
            if df is not None:
                data_placeholder.write(df)

        # Draw ROI box
        if ROI_TUPLE != DEFAULT_ROI or True:
            draw_box(
                draw,
                ROI_TUPLE,
                pil_image.width,
                pil_image.height,
                text="ROI",
                color=GREEN,
            )

        image_placeholder.image(np.array(pil_image),
                                caption="Processed image", use_column_width=True,)
        if stream is None:
            break
