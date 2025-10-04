# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
import time

sys.path.append("/home/gdli7/SparkAgent/fluxchat")
sys.path.append("/home/gdli7/SparkAgent/ComfyUI")

import base64
import copy
import io
import os, csv, sys
import random
import re
import string
import threading

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from addict import Dict
from chat_utils import get_md5
from importlib.metadata import version

refresh_sty = '\U0001f504'  # 🔄
clear_sty = '\U0001f5d1'  # 🗑️
upload_sty = '\U0001f5bc'  # 🖼️
sync_sty = '\U0001f4be'  # 💾
chat_sty = '\U0001F4AC'  # 💬

# lock = threading.Lock()
from acepp import run_acepp


class ChatBotUI(object):
    def __init__(self):
        # 参数配置
        self.model_choices = ["Editid ACEpp", "Kerachat", "Gemini 2.0 Flash Experimental "]
        self.default_model_name = "Editid ACEpp"
        self.pipe = Dict()

        self.max_msgs = 20
        self.gradio_version = version('gradio')

        self.cache_dir = "/home/gdli7/SparkAgent/results/"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def create_ui(self):
        css = '.chatbot.prose.md {opacity: 1.0 !important} #chatbot {opacity: 1.0 !important}'
        with gr.Blocks(css=css, title='Chatbot', head='Chatbot', analytics_enabled=False):
            gr.Markdown("""# Fluxchat人物版本""")
            with gr.Accordion("NOTE", open=True):
                v12_textbox = gr.Textbox(
                    value="根据人像生成需要结合reference模式，对生成图片可以不断通过提示此修改",
                    label="", lines=1)

            self.history = gr.State(value=[])
            self.images = gr.State(value={})
            self.history_result = gr.State(value={})
            self.retry_msg = gr.State(value='')

            with gr.Group():
                self.ui_mode = gr.State(value='chatbot')
                with gr.Row(equal_height=True, visible=True) as self.chat_group:
                    with gr.Column(visible=True) as self.chat_page:
                        self.chatbot = gr.Chatbot(
                            height=600,
                            value=[],
                            bubble_full_width=False,
                            show_copy_button=True,
                            container=False,
                            placeholder='<strong>Chat Box</strong>')
                        with gr.Row():
                            self.clear_btn = gr.Button(clear_sty + ' Clear Chat', size='sm')

                    with gr.Column(visible=False) as self.editor_page:
                        with gr.Tabs(visible=False) as self.upload_tabs:
                            with gr.Tab(id='ImageUploader', label='Image Uploader', visible=True) as self.upload_tab:
                                self.image_uploader = gr.Image(
                                    height=550,
                                    interactive=True,
                                    type='pil',
                                    image_mode='RGB',
                                    sources=['upload'],
                                    elem_id='image_uploader',
                                    format='png')
                                with gr.Row():
                                    self.sub_btn_1 = gr.Button(value='Submit', elem_id='upload_submit')
                                    self.ext_btn_1 = gr.Button(value='Exit')

                        with gr.Tabs(visible=False) as self.edit_tabs:
                            with gr.Tab(id='ImageEditor', label='Image Editor') as self.edit_tab:
                                self.mask_type = gr.Dropdown(
                                    label='Mask Type',
                                    choices=['Background', 'Composite', 'Outpainting'], value='Background')
                                self.mask_type_info = gr.HTML(
                                    value=
                                    "<div style='background-color: white; padding-left: 15px; color: grey;'>Background mode will not erase the visual content in the mask area</div>"
                                )
                                with gr.Accordion(
                                        label='Outpainting Setting', open=True, visible=False) as self.outpaint_tab:
                                    with gr.Row(variant='panel'):
                                        self.top_ext = gr.Slider(
                                            show_label=True,
                                            label='Top Extend Ratio',
                                            minimum=0.0,
                                            maximum=2.0,
                                            step=0.1,
                                            value=0.25)
                                        self.bottom_ext = gr.Slider(
                                            show_label=True,
                                            label='Bottom Extend Ratio',
                                            minimum=0.0,
                                            maximum=2.0,
                                            step=0.1,
                                            value=0.25)
                                    with gr.Row(variant='panel'):
                                        self.left_ext = gr.Slider(
                                            show_label=True,
                                            label='Left Extend Ratio',
                                            minimum=0.0,
                                            maximum=2.0,
                                            step=0.1,
                                            value=0.25)
                                        self.right_ext = gr.Slider(
                                            show_label=True,
                                            label='Right Extend Ratio',
                                            minimum=0.0,
                                            maximum=2.0,
                                            step=0.1,
                                            value=0.25)
                                    with gr.Row(variant='panel'):
                                        self.img_pad_btn = gr.Button(value='Pad Image')

                                self.image_editor = gr.ImageMask(
                                    value=None,
                                    sources=[],
                                    layers=False,
                                    label='Edit Image',
                                    elem_id='image_editor',
                                    format='png')
                                with gr.Row():
                                    self.sub_btn_2 = gr.Button(value='Submit', elem_id='edit_submit')
                                    self.ext_btn_2 = gr.Button(value='Exit')

                            with gr.Tab(id='ImageViewer', label='Image Viewer') as self.image_view_tab:
                                self.image_viewer = gr.Image(
                                    label='Image',
                                    type='pil',
                                    show_download_button=True,
                                    elem_id='image_viewer')
                                self.ext_btn_3 = gr.Button(value='Exit')

                with gr.Row(equal_height=True, visible=False) as self.legacy_group:
                    with gr.Column():
                        self.legacy_image_uploader = gr.Image(
                            height=550,
                            interactive=True,
                            type='pil',
                            image_mode='RGB',
                            elem_id='legacy_image_uploader',
                            format='png')
                    with gr.Column():
                        self.legacy_image_viewer = gr.Image(
                            label='Image',
                            height=550,
                            type='pil',
                            interactive=False,
                            show_download_button=True,
                            elem_id='image_viewer')

                with gr.Accordion(label='Setting', open=False):
                    with gr.Row():
                        self.model_name_dd = gr.Dropdown(
                            choices=self.model_choices,
                            value=self.default_model_name,
                            label='Model Version')

                    with gr.Row():
                        with gr.Column(scale=8, min_width=500):
                            with gr.Row():
                                self.step = gr.Slider(minimum=1,
                                                      maximum=1000,
                                                      value=20,
                                                      label='Sample Step')
                                self.cfg_scale = gr.Slider(
                                    minimum=1.0,
                                    maximum=100.0,
                                    value=50,
                                    label='Guidance Scale')

                                self.seed = gr.Slider(minimum=-1,
                                                      maximum=10000000,
                                                      value=-1,
                                                      label='Seed')
                                self.output_height = gr.Slider(
                                    minimum=256,
                                    maximum=1440,
                                    value=576,
                                    label='reference Height')
                                self.output_width = gr.Slider(
                                    minimum=256,
                                    maximum=1440,
                                    value=1024,
                                    label='reference Width')
                        with gr.Column(scale=1, min_width=50):
                            self.use_history = gr.Checkbox(value=False, label='Use History')

                with gr.Row(variant='panel', equal_height=True, show_progress=False):
                    with gr.Column(scale=1, min_width=100, visible=True) as self.upload_panel:
                        self.upload_btn = gr.Button(value=upload_sty + ' Upload', variant='secondary')
                    with gr.Column(scale=5, min_width=500):
                        self.text = gr.Textbox(placeholder='Input "@" find history of image',
                                               label='Instruction', container=False)
                    with gr.Column(scale=1, min_width=100):
                        self.chat_btn = gr.Button(value='Generate', variant='primary')
                    with gr.Column(scale=1, min_width=100):
                        self.retry_btn = gr.Button(value=refresh_sty + ' Retry', variant='secondary')
                    with gr.Column(scale=1, min_width=100):
                        self.mode_checkbox = gr.Checkbox(value=True, label='ChatBot')
                    with gr.Column(scale=1, min_width=100):
                        self.reference_checkbox = gr.Checkbox(value=False, label='Reference')

                with gr.Row():
                    self.gallery = gr.Gallery(visible=True,
                                              label='History',
                                              columns=10,
                                              allow_preview=False,
                                              interactive=False)

    def set_callbacks(self, *args, **kwargs):
        # chatbox的转换 -------------------------------
        def mode_change(mode_check):
            if mode_check:
                # ChatBot
                return (
                    gr.Row(visible=False),
                    gr.Row(visible=True),
                    gr.Button(value='Generate'),
                    gr.State(value='chatbot'),
                    gr.Column(visible=True)
                )
            else:
                # Legacy
                return (
                    gr.Row(visible=True),
                    gr.Row(visible=False),
                    gr.Button(value=chat_sty + ' Chat'),  # chat模式
                    gr.State(value='legacy'),
                    gr.Column(visible=False)
                )

        self.mode_checkbox.change(mode_change, inputs=[self.mode_checkbox],
                                  outputs=[self.legacy_group,
                                           self.chat_group,
                                           self.chat_btn,
                                           self.ui_mode,
                                           self.upload_panel, ])

        # ------------------- ChatBot -------------------
        # import pdb;pdb.set_trace()
        def generate_gallery(text, images):
            """
            输入text，就会从images中找图片
            :param text:
            :param images:
            :return:
            """
            if text.endswith(' '):
                return gr.update(), gr.update(visible=True)
            elif text.endswith('@'):
                gallery_info = []
                for image_id, image_meta in images.items():
                    thumbnail_path = image_meta['thumbnail']
                    gallery_info.append((thumbnail_path, image_id))
                return gr.update(), gr.update(visible=True, value=gallery_info)
            else:
                gallery_info = []
                match = re.search('@([^@ ]+)$', text)
                if match:
                    prefix = match.group(1)
                    for image_id, image_meta in images.items():
                        if not image_id.startswith(prefix):
                            continue
                        thumbnail_path = image_meta['thumbnail']
                        gallery_info.append((thumbnail_path, image_id))

                    if len(gallery_info) > 0:
                        return gr.update(), gr.update(visible=True, value=gallery_info)
                    else:
                        return gr.update(), gr.update(visible=True)
                else:
                    return gr.update(), gr.update(visible=True)

        self.text.input(generate_gallery,
                        inputs=[self.text, self.images],
                        outputs=[self.text, self.gallery],
                        # show_progress='hidden'
                        )

        # ------------------- Legacy -------------------
        def select_image(text, evt: gr.SelectData):
            """
            选择图片时把image_id带入text
            :param text:
            :param evt:
            :return:
            """
            image_id = evt.value['caption']
            text = '@'.join(text.split('@')[:-1]) + f'@{image_id} '
            return gr.update(value=text), gr.update(visible=True, value=None)

        self.gallery.select(select_image,
                            inputs=self.text,
                            outputs=[self.text, self.gallery])

        # ------------------------------- run chat -------------------------------
        def run_chat(
                message,
                legacy_image,
                ui_mode,
                history,
                images,
                use_history,
                history_result,
                cfg_scale,
                step,
                seed,
                output_h,
                output_w,
                model_name,
                progress=gr.Progress(track_tqdm=True)):
            # import pdb; pdb.set_trace()
            legacy_img_ids = []
            if ui_mode == 'legacy':
                if legacy_image is not None:
                    history, images, img_id = self.add_uploaded_image_to_history(
                        legacy_image, history, images)
                    legacy_img_ids.append(img_id)

            retry_msg = message
            gen_id = get_md5(message)[:12]
            save_path = os.path.join(self.cache_dir, f'{gen_id}.png')

            img_ids = re.findall('@(.*?)[ ,;.?$]', message)
            history_io = None

            if len(img_ids) < 1:
                img_ids = legacy_img_ids
                for img_id in img_ids:
                    if f'@{img_id}' not in message:
                        message = f'@{img_id} ' + message

            new_message = message

            if len(img_ids) > 0:
                edit_image, edit_image_mask, edit_task = [], [], []
                for i, img_id in enumerate(img_ids):  # images是chatbox中的
                    if img_id not in images:
                        gr.Info(
                            f'The input image ID {img_id} is not exist... Skip loading image.'
                        )
                        continue
                    # placeholder = '{image}' if i == 0 else '{' + f'image{i}' + '}'
                    # if placeholder not in new_message:
                    #     new_message = re.sub(f'@{img_id}', placeholder,
                    #                          new_message)
                    # else:
                    #     new_message = re.sub(f'@{img_id} ', "",
                    #                          new_message, 1)
                    img_meta = images[img_id]
                    img_path = img_meta['image']
                    img_mask = img_meta['mask']
                    img_mask_type = img_meta['mask_type']
                    if img_mask_type is not None and img_mask_type == 'Composite':
                        task = 'inpainting'
                    else:
                        task = ''
                    edit_image.append(Image.open(img_path).convert('RGB'))
                    edit_image_mask.append(
                        Image.open(img_mask).
                        convert('L') if img_mask is not None else None)
                    edit_task.append(task)

                    if use_history and (img_id in history_result):
                        history_io = history_result[img_id]

                buffered = io.BytesIO()
                edit_image[0].save(buffered, format='PNG')
                img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                img_str = f'<img src="data:image/png;base64,{img_b64}" style="pointer-events: none;">'
                pre_info = f'Received one or more images, so image editing is conducted.\n The first input image @{img_ids[0]} is:\n {img_str}'
            else:
                pre_info = 'No image ids were found in the provided text prompt, so text-guided image generation is conducted. \n'
                edit_image = None
                edit_image_mask = None
                edit_task = ''
            # if new_message == "":
            #     new_message = "a beautiful girl wear a skirt."
            # print(new_message)

            imgs = run_acepp(
                image=edit_image[0],
                prompt=new_message,
                output_height=output_h,
                output_width=output_w,
                sample_steps=step,
                guide_scale=cfg_scale,
                seed=seed,
            )

            img = imgs[0]
            img.save(save_path, format='PNG')

            if history_io: # 不勾选history没啥用的
                history_io_new = copy.deepcopy(history_io)
                history_io_new['image'] += edit_image[:1]
                history_io_new['mask'] += edit_image_mask[:1]
                history_io_new['task'] += edit_task[:1]
                history_io_new['prompt'] += [new_message]
                history_io_new['image'] = history_io_new['image'][-5:]
                history_io_new['mask'] = history_io_new['mask'][-5:]
                history_io_new['task'] = history_io_new['task'][-5:]
                history_io_new['prompt'] = history_io_new['prompt'][-5:]
                history_result[gen_id] = history_io_new
            elif edit_image is not None and len(edit_image) > 0:
                history_io_new = {
                    'image': edit_image[:1],
                    'mask': edit_image_mask[:1],
                    'task': edit_task[:1],
                    'prompt': [new_message]
                }
                history_result[gen_id] = history_io_new

            w, h = img.size
            if w > h:
                tb_w = 128
                tb_h = int(h * tb_w / w)
            else:
                tb_h = 128
                tb_w = int(w * tb_h / h)

            thumbnail_path = os.path.join(self.cache_dir,
                                          f'{gen_id}_thumbnail.jpg')
            thumbnail = img.resize((tb_w, tb_h))
            thumbnail.save(thumbnail_path, format='JPEG')

            images[gen_id] = {
                'image': save_path,
                'mask': None,
                'mask_type': None,
                'thumbnail': thumbnail_path
            }

            buffered = io.BytesIO()
            img.convert('RGB').save(buffered, format='JPEG')
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            img_str = f'<img src="data:image/jpg;base64,{img_b64}" style="pointer-events: none;">'

            history.append(
                (message,
                 f'{pre_info} The generated image @{gen_id} is:\n {img_str}'))

            while len(history) >= self.max_msgs:
                history.pop(0)

            return (history,
                    images,
                    gr.Image(value=save_path),
                    history_result,
                    self.get_history(history),
                    gr.update(),
                    gr.update(visible=True),
                    retry_msg)

        chat_inputs = [
            self.legacy_image_uploader,
            self.ui_mode,
            self.history,
            self.images,
            self.use_history,
            self.history_result,
            self.cfg_scale,
            self.step,
            self.seed,
            self.output_height,
            self.output_width,
            self.model_name_dd
        ]

        chat_outputs = [
            self.history,
            self.images,
            self.legacy_image_viewer,
            self.history_result,
            self.chatbot,
            self.text,
            self.gallery,
            self.retry_msg
        ]

        self.chat_btn.click(run_chat,
                            inputs=[self.text] + chat_inputs,
                            outputs=chat_outputs)

        self.text.submit(run_chat,
                         inputs=[self.text] + chat_inputs,
                         outputs=chat_outputs)

        def retry_fn(*args):
            return run_chat(*args)

        self.retry_btn.click(retry_fn,
                             inputs=[self.retry_msg] + chat_inputs,
                             outputs=chat_outputs)

        # --------------------------------------------------------------
        def upload_image():
            return (gr.update(visible=True, scale=1),
                    gr.update(visible=True, scale=1),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    # gr.update(visible=False),
                    gr.update(visible=True))

        self.upload_btn.click(upload_image,
                              inputs=[],
                              outputs=[
                                  self.chat_page,
                                  self.editor_page,
                                  self.upload_tab,
                                  self.edit_tab,
                                  self.image_view_tab,
                                  # self.video_view_tab,
                                  self.upload_tabs
                              ])

        # ---------------------------------------------------------------
        def edit_image(evt: gr.SelectData):
            """
            从chatbox中点击图片返回编辑区域
            :param evt:
            :return:
            """
            if isinstance(evt.value, str):
                img_b64s = re.findall(
                    '<img src="data:image/png;base64,(.*?)" style="pointer-events: none;">',
                    evt.value)
                imgs = [
                    Image.open(io.BytesIO(base64.b64decode(copy.deepcopy(i))))
                    for i in img_b64s
                ]
                if len(imgs) > 0:
                    if len(imgs) == 2:
                        if self.gradio_version >= '5.0.0':
                            view_img = copy.deepcopy(imgs[-1])
                        else:
                            view_img = copy.deepcopy(imgs)
                        edit_img = copy.deepcopy(imgs[-1])
                    else:
                        if self.gradio_version >= '5.0.0':
                            view_img = copy.deepcopy(imgs[-1])
                        else:
                            view_img = [
                                copy.deepcopy(imgs[-1]),
                                copy.deepcopy(imgs[-1])
                            ]
                        edit_img = copy.deepcopy(imgs[-1])

                    return (gr.update(visible=True, scale=1),
                            gr.update(visible=True, scale=1),
                            gr.update(visible=False),
                            gr.update(visible=True),
                            gr.update(visible=True),
                            gr.update(value=edit_img),
                            gr.update(value=view_img),
                            gr.update(visible=True))
                else:
                    return (gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update())
            else:
                return (gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update())

        self.chatbot.select(edit_image,
                            outputs=[
                                self.chat_page,
                                self.editor_page,
                                self.upload_tab,
                                self.edit_tab,
                                self.image_view_tab,
                                self.image_editor,
                                self.image_viewer,
                                self.edit_tabs
                            ])

        if self.gradio_version < '5.0.0':
            self.image_viewer.change(lambda x: x,
                                     inputs=self.image_viewer,
                                     outputs=self.image_viewer)

        #----------------------------------------------------------------------------
        def submit_upload_image(image, history, images):
            """
            图片上传的submit
            :param image:
            :param history:
            :param images:
            :return:
            """
            history, images, _ = self.add_uploaded_image_to_history(
                image, history, images)
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(value=self.get_history(history)),
                    history,
                    images)

        self.sub_btn_1.click(
            submit_upload_image,
            inputs=[self.image_uploader, self.history, self.images],
            outputs=[
                self.editor_page,
                self.chat_page,
                self.chatbot,
                self.history,
                self.images
            ])

        ########################################
        def submit_edit_image(imagemask, mask_type, history, images):
            history, images = self.add_edited_image_to_history(
                imagemask, mask_type, history, images)
            return gr.update(visible=False), gr.update(
                visible=True), gr.update(
                value=self.get_history(history)), history, images

        self.sub_btn_2.click(submit_edit_image,
                             inputs=[
                                 self.image_editor, self.mask_type,
                                 self.history, self.images
                             ],
                             outputs=[
                                 self.editor_page, self.chat_page,
                                 self.chatbot, self.history, self.images
                             ])

        ########################################
        def exit_edit():
            return gr.update(visible=False), gr.update(visible=True, scale=3)

        self.ext_btn_1.click(exit_edit,
                             outputs=[self.editor_page, self.chat_page])
        self.ext_btn_2.click(exit_edit,
                             outputs=[self.editor_page, self.chat_page])
        self.ext_btn_3.click(exit_edit,
                             outputs=[self.editor_page, self.chat_page])

        ########################################
        def update_mask_type_info(mask_type):
            if mask_type == 'Background':
                info = 'Background mode will not erase the visual content in the mask area'
                visible = False
            elif mask_type == 'Composite':
                info = 'Composite mode will erase the visual content in the mask area'
                visible = False
            elif mask_type == 'Outpainting':
                info = 'Outpaint mode is used for preparing input image for outpainting task'
                visible = True
            return (gr.update(
                visible=True,
                value=
                f"<div style='background-color: white; padding-left: 15px; color: grey;'>{info}</div>"
            ), gr.update(visible=visible))

        self.mask_type.change(update_mask_type_info,
                              inputs=self.mask_type,
                              outputs=[self.mask_type_info, self.outpaint_tab])

        ########################################
        def extend_image(top_ratio, bottom_ratio, left_ratio, right_ratio,
                         image):
            img = cv2.cvtColor(image['background'], cv2.COLOR_RGBA2RGB)
            h, w = img.shape[:2]
            new_h = int(h * (top_ratio + bottom_ratio + 1))
            new_w = int(w * (left_ratio + right_ratio + 1))
            start_h = int(h * top_ratio)
            start_w = int(w * left_ratio)
            new_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            new_mask = np.ones((new_h, new_w, 1), dtype=np.uint8) * 255
            new_img[start_h:start_h + h, start_w:start_w + w, :] = img
            new_mask[start_h:start_h + h, start_w:start_w + w] = 0
            layer = np.concatenate([new_img, new_mask], axis=2)
            value = {
                'background': new_img,
                'composite': new_img,
                'layers': [layer]
            }
            return gr.update(value=value)

        self.img_pad_btn.click(extend_image,
                               inputs=[
                                   self.top_ext, self.bottom_ext,
                                   self.left_ext, self.right_ext,
                                   self.image_editor
                               ],
                               outputs=self.image_editor)

        ########################################
        def clear_chat(history, images, history_result):
            history.clear()
            images.clear()
            history_result.clear()
            return history, images, history_result, self.get_history(history)

        self.clear_btn.click(
            clear_chat,
            inputs=[self.history, self.images, self.history_result],
            outputs=[
                self.history, self.images, self.history_result, self.chatbot
            ])

    def get_history(self, history):
        info = []
        for item in history:
            new_item = [None, None]
            if isinstance(item[0], str) and item[0].endswith('.mp4'):
                new_item[0] = gr.Video(item[0], format='mp4')
            else:
                new_item[0] = item[0]
            if isinstance(item[1], str) and item[1].endswith('.mp4'):
                new_item[1] = gr.Video(item[1], format='mp4')
            else:
                new_item[1] = item[1]
            info.append(new_item)
        return info

    def generate_random_string(self, length=20):
        letters_and_digits = string.ascii_letters + string.digits
        random_string = ''.join(
            random.choice(letters_and_digits) for i in range(length))
        return random_string

    def add_edited_image_to_history(self, image, mask_type, history, images):
        if mask_type == 'Composite':
            img = Image.fromarray(image['composite'])
        else:
            img = Image.fromarray(image['background'])

        img_id = get_md5(self.generate_random_string())[:12]
        save_path = os.path.join(self.cache_dir, f'{img_id}.png')
        img.convert('RGB').save(save_path)

        mask = image['layers'][0][:, :, 3]
        mask = Image.fromarray(mask).convert('RGB')
        mask_path = os.path.join(self.cache_dir, f'{img_id}_mask.png')
        mask.save(mask_path)

        w, h = img.size
        if w > h:
            tb_w = 128
            tb_h = int(h * tb_w / w)
        else:
            tb_h = 128
            tb_w = int(w * tb_h / h)

        if mask_type == 'Background':
            comp_mask = np.array(mask, dtype=np.uint8)
            mask_alpha = (comp_mask[:, :, 0:1].astype(np.float32) *
                          0.6).astype(np.uint8)
            comp_mask = np.concatenate([comp_mask, mask_alpha], axis=2)
            thumbnail = Image.alpha_composite(
                img.convert('RGBA'),
                Image.fromarray(comp_mask).convert('RGBA')).convert('RGB')
        else:
            thumbnail = img.convert('RGB')

        thumbnail_path = os.path.join(self.cache_dir,
                                      f'{img_id}_thumbnail.jpg')
        thumbnail = thumbnail.resize((tb_w, tb_h))
        thumbnail.save(thumbnail_path, format='JPEG')

        buffered = io.BytesIO()
        img.convert('RGB').save(buffered, format='PNG')
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_str = f'<img src="data:image/png;base64,{img_b64}" style="pointer-events: none;">'

        buffered = io.BytesIO()
        mask.convert('RGB').save(buffered, format='PNG')
        mask_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        mask_str = f'<img src="data:image/png;base64,{mask_b64}" style="pointer-events: none;">'

        images[img_id] = {
            'image': save_path,
            'mask': mask_path,
            'mask_type': mask_type,
            'thumbnail': thumbnail_path
        }
        history.append((
            None,
            f'This is edited image and mask:\n {img_str} {mask_str} image ID is: {img_id}'
        ))
        return history, images

    def add_uploaded_image_to_history(self, img, history, images):
        """
        submit同时保存了原图和所旅途，并未进行尺寸处理
        :param img:
        :param history:
        :param images:
        :return:
        """
        # import pdb;pdb.set_trace()
        img_id = get_md5(self.generate_random_string())[:12]
        save_path = os.path.join(self.cache_dir, f'{img_id}.png')
        # 对超过2048的图做一些处理
        w, h = img.size
        if w > 2048:
            ratio = w / 2048.
            w = 2048
            h = int(h / ratio)
        if h > 2048:
            ratio = h / 2048.
            h = 2048
            w = int(w / ratio)
        img = img.resize((w, h))
        img.save(save_path)

        # 做了一个缩略图
        w, h = img.size
        if w > h:
            tb_w = 128
            tb_h = int(h * tb_w / w)
        else:
            tb_h = 128
            tb_w = int(w * tb_h / h)
        thumbnail_path = os.path.join(self.cache_dir,
                                      f'{img_id}_thumbnail.jpg')
        thumbnail = img.resize((tb_w, tb_h))
        thumbnail.save(thumbnail_path, format='JPEG')

        images[img_id] = {
            'image': save_path,
            'mask': None,
            'mask_type': None,
            'thumbnail': thumbnail_path
        }

        buffered = io.BytesIO()
        img.convert('RGB').save(buffered, format='PNG')
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_str = f'<img src="data:image/png;base64,{img_b64}" style="pointer-events: none;">'

        history.append(
            (None,
             f'This is uploaded image:\n {img_str} image ID is: {img_id}'))
        return history, images, img_id


if __name__ == '__main__':
    with gr.Blocks() as demo:
        chatbot = ChatBotUI()
        chatbot.create_ui()
        chatbot.set_callbacks()
    demo.launch(server_name="0.0.0.0", server_port=9005, allowed_paths=["/home/gdli7/SparkAgent/results"])
