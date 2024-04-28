import os,glob,sys
import torch
import torch.utils.data as data
import numpy as np
import cv2
import json
import string

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from os.path import join as ospj
from torchvision.utils import save_image
from random import shuffle, randint, choice, choices
from fontTools.ttLib import TTFont

sys.path.append(os.path.dirname(__file__))
from utils.augment import aug_method_color_texture


def get_font_paths(root):

    def traversal_files(dir):

        file_paths = []
        for item in os.scandir(dir):
            if item.is_file():
                file_paths.append(item.path)
            else:
                file_paths += traversal_files(item)
        
        return file_paths

    font_paths = traversal_files(root)
    font_paths = [path for path in font_paths if path[-3:] in ('ttf', 'otf', 'TTF', 'OTF')]

    return font_paths


def get_char_dict():

    # Common Chinese characters
    with open("./dataset/utils/chars/GB2312_1.txt", "r") as fp:
        common_cn_chars = list(fp.readline()) # 3755
    # Complex Chinese characters
    with open("./dataset/utils/chars/GB2312_2.txt", "r") as fp:
        complex_cn_chars = list(fp.readline()) # 3008

    # cn_chars = [chr(i) for i in range(0x4e00, 0x9fa5 + 1)] # All Chinese characters
    cn_chars = common_cn_chars + complex_cn_chars # All Chinese characters
    hf_chars = [chr(i) for i in range(0xff00, 0xffef + 1)] # Halfwidth and Fullwidth Forms
    
    all_chars = list(string.printable) + cn_chars + hf_chars

    print(f"Size of common_cn_char set: {len(common_cn_chars)}")
    print(f"Size of complex_cn_char set: {len(complex_cn_chars)}")
    print(f"Size of all_char set: {len(all_chars)}")

    char_dict = {
        "common": common_cn_chars,
        "complex": complex_cn_chars,
        "all": all_chars
    }

    return char_dict

def get_char_list_from_ttf(font_file):

    f_obj = TTFont(font_file)
    m_dict = f_obj.getBestCmap()
    
    char_list = []
    for key, _ in m_dict.items():
        try:
            char_list.append(chr(key))
        except:
            continue

    return char_list


class CTWGlyphDataset(data.Dataset):

    def __init__(self, cfgs, datype) -> None:
        super().__init__()

        # basic
        self.type = datype
        self.editing = cfgs.editing
        if datype=="train": self.editing=False # editing only in test stage

        # constraint
        self.min_seg_ratio = cfgs.min_seg_ratio
        self.max_seg_aspect = cfgs.max_seg_aspect
        self.seq_len = cfgs.seq_len
        self.std_font_path = cfgs.std_font_path
        self.ref_size = cfgs.ref_size
        self.n_ref_style = cfgs.n_ref_style

        # path
        data_root = ospj(cfgs.data_root, "image")
        cache_path = ospj(cfgs.data_root, f"{datype}_precache.pth")
        self.char_dict = get_char_dict()

        if self.editing: cfgs.refresh = True
        if os.path.exists(cache_path) and not cfgs.refresh:
            self.items = torch.load(cache_path)
        else:
            anno_path = ospj(cfgs.data_root, "annotation", f"{datype}.jsonl")
            with open(anno_path) as fp:
                lines = fp.readlines()
            annos = [json.loads(line) for line in lines]

            self.items = []
            total_count = 0
            for anno in tqdm(annos):
                image_path = ospj(data_root, anno["file_name"])
                h, w = anno["height"], anno["width"]
                sens = anno["annotations"]
                for sen in sens:
                    total_count += 1
                    label = ""
                    seg_bboxs = []
                    for ins in sen:
                        if "occluded" in ins["attributes"]: continue
                        seg_bbox = np.array(ins["polygon"], dtype=np.float32)
                        seg_w = np.max(seg_bbox[:,0]) - np.min(seg_bbox[:,0])
                        seg_h = np.max(seg_bbox[:,1]) - np.min(seg_bbox[:,1])
                        if (seg_w/seg_h) > self.max_seg_aspect or (seg_h/seg_w) > self.max_seg_aspect: continue
                        label += ins["text"]
                        seg_bboxs.append(seg_bbox)

                    if len(label) == 0 or len(label) > self.seq_len: continue
                    seg_bboxs = np.stack(seg_bboxs, axis=0)
                    if np.any(seg_bboxs<0) or np.any(seg_bboxs[...,0]>w) or np.any(seg_bboxs[...,1]>h): continue

                    seg_bboxs = self.reorder_bbox(seg_bboxs)
                    mask_bbox, mask_area, seg_ratio = self.premeasure(h, w, seg_bboxs)
                    if seg_ratio < self.min_seg_ratio: continue

                    image = Image.open(image_path).convert("RGB")
                    image = np.asarray(image)

                    if self.editing:
                        label = "".join(choices(self.char_dict["common"], k=len(label)))
                    
                    c_ref = self.get_c_ref(label)
                    s_ref = self.get_s_ref(image, seg_bboxs)

                    item = {
                        "c_ref": c_ref,
                        "s_ref": s_ref
                    }

                    self.items.append(item)

            print(f"Data filtering completed. {len(self.items)} of {total_count} in total.")
            torch.save(self.items, cache_path)

        self.freq = cfgs.freq
        self.length = len(self.items) * self.freq
    
    def __len__(self):
        
        return self.length
    
    def reorder_bbox(self, bboxs):

        for i, bbox in enumerate(bboxs):

            sort_bbox = bbox[np.argsort(bbox[:,0]), :]
            left, right = np.split(sort_bbox, 2, axis=0)

            left = left[np.argsort(left[:,1]), :]
            right = right[np.argsort(right[:,1]), :]

            bbox = np.stack((left[0], right[0], right[1], left[1]))
            bboxs[i] = bbox
        
        return bboxs
    
    def premeasure(self, h:int, w:int, seg_bboxs:np.array):

        mask_rect = cv2.minAreaRect(seg_bboxs.reshape((-1,2)))
        mask_bbox = cv2.boxPoints(mask_rect).astype(np.int32)
        mask_area = cv2.contourArea(mask_bbox)
        mask_ratio = mask_area / (h*w)
        seg_ratio = mask_ratio / seg_bboxs.shape[0]

        return mask_bbox, mask_area, seg_ratio

    def get_c_ref(self, label):

        font = ImageFont.truetype(self.std_font_path, self.ref_size)

        c_refs = []
        for ch in label:

            std_l, std_t, std_r, std_b = font.getbbox(ch)
            std_h, std_w = std_b - std_t, std_r - std_l

            c_ref = Image.new(mode='L', size=(std_w, std_h), color=255)
            draw = ImageDraw.Draw(c_ref)
            draw.text((0, 0), ch, fill=0, font=font, anchor="lt")

            c_ref = np.asarray(c_ref)
            d_top, d_bottom, d_left, d_right = 0, 0, 0, 0
            if std_h >= std_w:
                d_left = d_right = (std_h-std_w)//2
            else:
                d_top = d_bottom = (std_w-std_h)//2

            c_ref = cv2.copyMakeBorder(c_ref, d_top, d_bottom, d_left, d_right, cv2.BORDER_CONSTANT, value=255)
            c_ref = cv2.resize(c_ref, (self.ref_size, self.ref_size))
            c_refs.append(c_ref)

        c_ref = np.stack(c_refs, axis=0).copy()
        c_ref = torch.from_numpy(c_ref).to(dtype=torch.float32) / 127.5 - 1.0

        return c_ref

    def get_seg_map(self, h, w, seg_bboxs):

        segs = []
        for seg_bbox in seg_bboxs:
            seg = np.zeros((h, w), dtype=np.uint8)
            seg = cv2.fillConvexPoly(seg, seg_bbox.astype(np.int32), color=1)
            seg = cv2.morphologyEx(seg, cv2.MORPH_DILATE, np.ones((3,3), np.int8), iterations=5) # dilate
            segs.append(seg)

        segs += [np.zeros_like(segs[0]) for i in range(self.seq_len-len(seg_bboxs))]
        seg_map = np.stack(segs, axis=0)

        return seg_map

    def get_s_ref(self, image, seg_bboxs):

        h, w = image.shape[:2]
        seg_map = self.get_seg_map(h, w, seg_bboxs)

        st, sb = (h-self.ref_size)//2, (w+self.ref_size)//2
        sl, sr = (h-self.ref_size)//2, (w+self.ref_size)//2
        std_bbox = np.array([[sl, st], [sr, st], [sr, sb], [sl, sb]], dtype=np.float32)

        s_refs = []
        for seg, seg_bbox in zip(seg_map, seg_bboxs):
            s_ref = image * seg[...,None]
            tran_mx = cv2.getAffineTransform(seg_bbox.astype(np.float32)[:3], std_bbox[:3])
            warped_s_ref = cv2.warpAffine(s_ref, tran_mx, (w, h))
            s_ref = warped_s_ref[st:sb, sl:sr, :]
            s_refs.append(s_ref)

        s_ref = np.stack(s_refs, axis=0).copy()
        s_ref = torch.from_numpy(s_ref.transpose(0,3,1,2)).to(dtype=torch.float32) / 127.5 - 1.0

        return s_ref
    
    def __getitem__(self, index):
        
        item = self.items[index//self.freq]
        c_refs = item["c_ref"]
        s_refs = item["s_ref"]

        len = c_refs.shape[0]
        idx = randint(0, len-1)
        c_ref = c_refs[idx][None]
        gt = s_refs[idx]

        s_ref = choices(s_refs, k=self.n_ref_style)
        s_ref = torch.stack(s_ref, dim=0)

        batch = {
            "c_ref": c_ref,
            "s_ref": s_ref,
            "gt": gt
        }

        return batch
    

class SynthGlyphDataset(data.Dataset):

    def __init__(self, cfgs, datype) -> None:
        super().__init__()

        # basic
        self.type = datype

        # constraint
        self.std_font_path = cfgs.std_font_path
        self.font_root = cfgs.font_root
        self.ref_size = cfgs.ref_size
        self.n_ref_style = cfgs.n_ref_style
        
        self.font_paths = get_font_paths(self.font_root)
        self.char_dict = get_char_dict()
        self.length = cfgs.length

    def __len__(self):

        return self.length
    
    def render_ch(self, chs, font_path):
        
        font =  ImageFont.truetype(font_path, self.ref_size)

        c_refs = []
        for ch in chs:
            std_l, std_t, std_r, std_b = font.getbbox(ch)
            std_h, std_w = std_b - std_t, std_r - std_l

            c_ref = Image.new(mode='L', size=(std_w, std_h), color=255)
            draw = ImageDraw.Draw(c_ref)
            draw.text((0, 0), ch, fill=0, font=font, anchor="lt")

            c_ref = np.asarray(c_ref)
            d_top, d_bottom, d_left, d_right = 0, 0, 0, 0
            if std_h >= std_w:
                d_left = d_right = (std_h-std_w)//2
            else:
                d_top = d_bottom = (std_w-std_h)//2

            c_ref = cv2.copyMakeBorder(c_ref, d_top, d_bottom, d_left, d_right, cv2.BORDER_CONSTANT, value=255)
            c_ref = cv2.resize(c_ref, (self.ref_size, self.ref_size))
            c_ref = np.stack([c_ref]*3, axis=-1)
            c_refs.append(c_ref)

        return c_refs


    def __getitem__(self, index):

        font_path = choice(self.font_paths)
        font_char_list = get_char_list_from_ttf(font_path)
        char_list = list(set(self.char_dict["common"]) & set(font_char_list))
        chs = choices(char_list, k=self.n_ref_style+1)

        seed = randint(0, 1e9)
        s_refs = self.render_ch(chs, font_path=font_path)
        aug_s_refs = [aug_method_color_texture(ref, seed, aug_bg=True) for ref in s_refs]

        c_gt = torch.from_numpy(s_refs[0].transpose(2,0,1)).to(dtype=torch.float32) / 127.5 - 1.0
        c_gt = c_gt.mean(dim=0, keepdim=True)
        s_gt = torch.from_numpy(aug_s_refs[0].transpose(2,0,1)).to(dtype=torch.float32) / 127.5 - 1.0

        c_ref = self.render_ch(chs[:1], font_path=self.std_font_path)[0]
        c_ref = torch.from_numpy(c_ref[...,:1].transpose(2,0,1)).to(dtype=torch.float32) / 127.5 - 1.0

        s_ref = np.stack(aug_s_refs[1:], axis=0)
        s_ref = torch.from_numpy(s_ref.transpose(0,3,1,2)).to(dtype=torch.float32) / 127.5 - 1.0

        batch = {
            "c_ref": c_ref,
            "s_ref": s_ref,
            "c_gt": c_gt,
            "s_gt": s_gt,
        }

        return batch