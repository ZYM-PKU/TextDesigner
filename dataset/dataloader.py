import os,glob,sys
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import cv2
import json

from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from os.path import join as ospj
from torchvision.utils import save_image
from random import randint, choices

sys.path.append(os.path.dirname(__file__))
from pre_dataloader import *


class ImageDataset(data.Dataset):

    def __init__(self, cfgs, datype) -> None:
        super().__init__()

        # basic
        self.type = datype
        self.editing = cfgs.editing
        if datype=="train": self.editing=False # editing only in test stage

        # constraint
        self.H = cfgs.H
        self.W = cfgs.W
        self.std_mask_ratio = cfgs.std_mask_ratio
        self.min_seg_ratio = cfgs.min_seg_ratio
        self.max_seg_aspect = cfgs.max_seg_aspect
        self.ref_size = cfgs.ref_size
        self.seq_len = cfgs.seq_len
        self.std_font_path = cfgs.std_font_path
        self.repeat = cfgs.repeat

        # memory
        self.count = 0
        self.char_dict = get_char_dict()

    def __len__(self):
        
        return self.length

    def get_seg_map(self, seg_bboxs):

        segs = []
        for seg_bbox in seg_bboxs:
            seg = np.zeros((self.H, self.W), dtype=np.uint8)
            seg = cv2.fillConvexPoly(seg, np.array(seg_bbox), color=1)
            seg = cv2.morphologyEx(seg, cv2.MORPH_DILATE, np.ones((3,3), np.int8), iterations=5) # dilate
            segs.append(seg)

        segs += [np.ones_like(segs[0]) for i in range(self.seq_len-len(seg_bboxs))]
        seg_map = np.stack(segs, axis=0)

        return seg_map

    def get_ctrl(self, label, seg_bboxs):

        ctrl_img = Image.new(mode='L', size=(self.W, self.H), color=0)
        draw = ImageDraw.Draw(ctrl_img)
        font = ImageFont.truetype(self.std_font_path, self.ref_size)

        for ch, seg_bbox in zip(label, seg_bboxs):
            
            std_l, std_t, std_r, std_b = font.getbbox(ch)
            std_h, std_w = std_b - std_t, std_r - std_l
            l, t = (self.W-std_w)//2, (self.H-std_h)//2

            font_image = Image.new(mode='RGBA', size=(self.W, self.H), color=(0,0,0,0))
            draw = ImageDraw.Draw(font_image)
            draw.text((l, t), ch, fill=(255,255,255,255), font=font, anchor="lt")
            font_image = np.asarray(font_image)

            std_bbox = np.array([[l, t], [l+std_w, t], [l+std_w, t+std_h], [l, t+std_h]], dtype=np.float32)
            tran_mx = cv2.getAffineTransform(std_bbox[:3], np.array(seg_bbox, dtype=np.float32)[:3])

            warped_font_image = cv2.warpAffine(font_image, tran_mx, (self.W, self.H))
            warped_font_image = Image.fromarray(warped_font_image)

            ctrl_img.paste(warped_font_image, mask=warped_font_image)
        
        ctrl_img = np.asarray(ctrl_img)[None].copy()

        return ctrl_img

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
        
        reference = np.stack(c_refs, axis=0).copy()

        return reference

    def get_s_ref(self, image, seg_map, seg_bboxs):

        st, sb, sl, sr = 0, self.ref_size, 0, self.ref_size
        std_bbox = np.array([[sl, st], [sr, st], [sr, sb], [sl, sb]], dtype=np.float32)

        s_refs = []
        tran_mxs = []
        for seg, seg_bbox in zip(seg_map, seg_bboxs):
            s_ref = image * seg[...,None]
            tran_mx = cv2.getAffineTransform(np.array(seg_bbox, dtype=np.float32)[:3], std_bbox[:3])
            warped_s_ref = cv2.warpAffine(s_ref, tran_mx, (self.ref_size, self.ref_size))
            s_refs.append(warped_s_ref)
            tran_mxs.append(tran_mx)

        reference = np.stack(s_refs, axis=0).copy()
        tran_mxs = np.stack(tran_mxs, axis=0).copy()

        return reference, tran_mxs

    def premeasure(self, h:int, w:int, seg_bboxs:np.array):

        mask_rect = cv2.minAreaRect(seg_bboxs.reshape((-1,2)))
        mask_bbox = cv2.boxPoints(mask_rect).astype(np.int32)
        mask_area = cv2.contourArea(mask_bbox)
        mask_ratio = mask_area / (h*w)
        seg_ratio = mask_ratio / seg_bboxs.shape[0]

        return mask_bbox, mask_area, seg_ratio
    
    def precalculate(self, h:int, w:int, seg_bboxs:np.array, mask_bbox:np.array, mask_area:float):

        # points in seg_bboxs must be in this order: 
        # 0 -> 1
        # |    |
        # 3 <- 2

        seg_ws, seg_hs = seg_bboxs[..., 0], seg_bboxs[..., 1]
        m_left, m_right = int(seg_ws.min()), int(seg_ws.max())
        m_top, m_bottom = int(seg_hs.min()), int(seg_hs.max())
        m_h, m_w = m_bottom-m_top, m_right-m_left

        ### augmentation
        if m_h > w or m_w > h: # expand
            aug_type = "expand"
            d_top, d_bottom, d_left, d_right = 0, 0, 0, 0
            if h >= w:
                d_left = randint(0, (h-w))
                d_right = (h-w) - d_left
                m_left, m_right = m_left + d_left, m_right + d_left
                seg_bboxs[..., 0] += d_left
                mask_bbox[..., 0] += d_left
            else:
                d_top = randint(0, (w-h))
                d_bottom = (w-h) - d_top
                m_top, m_bottom = m_top + d_top, m_bottom + d_top
                seg_bboxs[..., 1] += d_top
                mask_bbox[..., 1] += d_top
            aug_rbbox = np.array((d_top, d_bottom, d_left, d_right))
            h, w = h+d_top+d_bottom, w+d_left+d_right
        
        else: # crop
            aug_type = "crop"
            if h >= w:
                std_h = std_w = w
                if mask_area / (w*w) < self.std_mask_ratio:
                    std_h = std_w = int((mask_area/self.std_mask_ratio)**0.5)
            else:
                std_h = std_w = h
                if mask_area / (h*h) < self.std_mask_ratio:
                    std_h = std_w = int((mask_area/self.std_mask_ratio)**0.5)
            
            std_h, std_w = max(std_h, max(m_h, m_w)), max(std_w, max(m_h, m_w))
            n_top = randint(max(0, m_bottom-std_h), min(m_top, h-std_h))
            n_left = randint(max(0, m_right-std_w), min(m_left, w-std_w))
            m_top, m_bottom = m_top - n_top, m_bottom - n_top
            m_left, m_right = m_left - n_left, m_right - n_left
            seg_bboxs[..., 0], seg_bboxs[..., 1] = seg_bboxs[..., 0]-n_left, seg_bboxs[..., 1]-n_top
            mask_bbox[..., 0], mask_bbox[..., 1] = mask_bbox[..., 0]-n_left, mask_bbox[..., 1]-n_top
            aug_rbbox = np.array((n_top, n_top+std_h, n_left, n_left+std_w))
            h, w = std_h, std_w
        
        mask_rbbox = np.array((m_top, m_bottom, m_left, m_right))
        
        ### scaling
        scale = self.H / h
        seg_bboxs = (seg_bboxs * scale).astype(np.int32)
        mask_bbox = (mask_bbox * scale).astype(np.int32)
        mask_rbbox = (mask_rbbox * scale).astype(np.int32)

        return aug_type, aug_rbbox, seg_bboxs, mask_bbox, mask_rbbox
    
    def preprocess(self, image, item):
        
        image = np.asarray(image)
        label = item["label"]
        aug_type, aug_rbbox = item["aug_type"], item["aug_rbbox"]
        seg_bboxs, mask_bbox = item["seg_bboxs"], item["mask_bbox"]
        mask_rbbox = item["mask_rbbox"]
        m_top, m_bottom, m_left, m_right = mask_rbbox

        ### augmentation
        if aug_type == "expand":
            d_top, d_bottom, d_left, d_right = aug_rbbox.tolist()
            image = cv2.copyMakeBorder(image, d_top, d_bottom, d_left, d_right, cv2.BORDER_REPLICATE)
        elif aug_type == "crop":
            n_top, n_bottom, n_left, n_right = aug_rbbox
            image = image[n_top:n_bottom, n_left:n_right, :]
        
        image = cv2.resize(image, (self.W, self.H))
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, np.array(mask_bbox), color=1)

        ### conditioning
        seg_map = self.get_seg_map(seg_bboxs)
        ctrl = self.get_ctrl(label, seg_bboxs)
        c_ref = self.get_c_ref(label)
        s_ref, tran_mxs = self.get_s_ref(image, seg_map, seg_bboxs)

        image = torch.from_numpy(image.transpose(2,0,1)).to(dtype=torch.float32) / 127.5 - 1.0
        mask = torch.from_numpy(mask[None]).to(dtype=torch.float32)
        masked = image * (1 - mask)

        seg_map = torch.from_numpy(seg_map)
        ctrl = torch.from_numpy(ctrl).to(dtype=torch.float32) / 127.5 - 1.0
        c_ref = torch.from_numpy(c_ref).to(dtype=torch.float32) / 127.5 - 1.0
        s_ref = torch.from_numpy(s_ref.transpose(0,3,1,2)).to(dtype=torch.float32) / 127.5 - 1.0
        tran_mxs = torch.from_numpy(tran_mxs)

        pad_len = self.seq_len-len(label)
        if pad_len > 0:
            c_pad = torch.zeros((pad_len, self.ref_size, self.ref_size)).to(c_ref)
            s_pad = torch.zeros((pad_len, 3, self.ref_size, self.ref_size)).to(s_ref)
            tran_pad = torch.eye(3)[None].tile(pad_len,1,1).to(tran_mxs)
            c_ref = torch.cat([c_ref, c_pad], dim=0)
            s_ref = torch.cat([s_ref, s_pad], dim=0)
            tran_mxs = torch.cat([tran_mxs, tran_pad[:,:2,:]], dim=0)
        
        return image, mask, masked, seg_map, ctrl, c_ref, s_ref, tran_mxs

    def __getitem__(self, index):

        item = self.items[index]

        ### load image
        image_path = item["image_path"]
        image = Image.open(image_path).convert("RGB")

        if self.editing: 
            item["label"] = "".join(choices(self.char_dict["common"], k=len(item["label"])))

        ### preprocess
        image, mask, masked, seg_map, ctrl, c_ref, s_ref, tran_mxs = self.preprocess(image, item)

        batch = {
            "name": str(self.count),
            "label": item["label"],
            "ref_len": len(item["label"]),
            "image": image,
            "mask": mask,
            "masked": masked,
            "seg_map": seg_map,
            "ctrl": ctrl,
            "c_ref": c_ref,
            "s_ref": s_ref,
            "s_tgt": s_ref,
            "tran_mxs": tran_mxs,
            "r_bbox": item["mask_rbbox"],
            "seg_mask": item["seg_mask"]
        }

        self.count += 1

        return batch


class TextLogoDataset(ImageDataset):

    def __init__(self, cfgs, datype) -> None:
        super().__init__(cfgs, datype)

        # path
        data_root = ospj(cfgs.data_root, datype)
        cache_path = ospj(cfgs.data_root, f"{datype}_cache.pth")

        if os.path.exists(cache_path) and not cfgs.refresh:
            self.items = torch.load(cache_path)
        else:
            anno_dict = {}
            anno_path = ospj(cfgs.data_root, "logo_titles.txt")
            with open(anno_path) as fp:
                annos = fp.readlines()
            for anno_line in annos:
                anno_lst = anno_line[:-1].split(" ")
                if len(anno_lst)>4: continue
                key = anno_lst[0]
                label = anno_lst[2]
                anno_dict[key] = label

            self.items = []
            item_paths = sorted(glob.glob(ospj(data_root, "*")))
            total_count = 0
            for _ in range(self.repeat):
                for item_path in tqdm(item_paths):
                    total_count +=1
                    
                    key = item_path.split(os.sep)[-1]
                    if not key in anno_dict: continue
                    label = anno_dict[key]

                    image_path = ospj(item_path, "poster.jpg")
                    image = Image.open(image_path).convert("RGB")
                    h, w = image.height, image.width

                    coords_det = np.load(ospj(item_path, "coords_det.npy"))
                    seg_bboxs = coords_det[:len(label)].reshape((-1,4,2)).astype(np.float32)

                    if len(label) == 0 or len(label) > self.seq_len: continue
                    if np.any(seg_bboxs<0) or np.any(seg_bboxs[...,0]>w) or np.any(seg_bboxs[...,1]>h): continue

                    seg_mask = np.array([1]*len(label) + [0]*(self.seq_len-len(label)))
                    mask_bbox, mask_area, seg_ratio = self.premeasure(h, w, seg_bboxs)
                    if seg_ratio < self.min_seg_ratio: continue

                    aug_type, aug_rbbox, seg_bboxs, mask_bbox, mask_rbbox = \
                        self.precalculate(h, w, seg_bboxs, mask_bbox, mask_area)

                    item = {
                        "image_path": image_path, 
                        "label": label,
                        "aug_type": aug_type,
                        "aug_rbbox": torch.tensor(aug_rbbox),
                        "seg_mask": torch.tensor(seg_mask),
                        "seg_bboxs": torch.tensor(seg_bboxs),
                        "mask_bbox": torch.tensor(mask_bbox),
                        "mask_rbbox": torch.tensor(mask_rbbox)
                    }

                    self.items.append(item)

            print(f"Data filtering completed. {len(self.items)} of {total_count} in total.")
            torch.save(self.items, cache_path)

        self.length = len(self.items)


class CTWDataset(ImageDataset):

    def __init__(self, cfgs, datype) -> None:
        super().__init__(cfgs, datype)

        # path
        data_root = ospj(cfgs.data_root, "image")
        cache_path = ospj(cfgs.data_root, f"{datype}_cache.pth")

        if os.path.exists(cache_path) and not cfgs.refresh:
            self.items = torch.load(cache_path)
        else:
            anno_path = ospj(cfgs.data_root, "annotation", f"{datype}.jsonl")
            with open(anno_path) as fp:
                lines = fp.readlines()
            annos = [json.loads(line) for line in lines]

            self.items = []
            total_count = 0
            for _ in range(self.repeat):
                for anno in tqdm(annos):
                    image_path = ospj(data_root, anno["file_name"])
                    h, w = anno["height"], anno["width"]
                    sens = anno["annotations"]
                    for sen in sens:
                        total_count += 1
                        label = ""
                        seg_bboxs = []
                        for ins in sen:
                            # if "occluded" in ins["attributes"]: continue
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
                        seg_mask = np.array([1]*len(label) + [0]*(self.seq_len-len(label)))
                        mask_bbox, mask_area, seg_ratio = self.premeasure(h, w, seg_bboxs)
                        if seg_ratio < self.min_seg_ratio: continue

                        aug_type, aug_rbbox, seg_bboxs, mask_bbox, mask_rbbox = \
                            self.precalculate(h, w, seg_bboxs, mask_bbox, mask_area)

                        item = {
                            "image_path": image_path, 
                            "label": label,
                            "aug_type": aug_type,
                            "aug_rbbox": torch.tensor(aug_rbbox),
                            "seg_mask": torch.tensor(seg_mask),
                            "seg_bboxs": torch.tensor(seg_bboxs),
                            "mask_bbox": torch.tensor(mask_bbox),
                            "mask_rbbox": torch.tensor(mask_rbbox)
                        }

                        self.items.append(item)

            print(f"Data filtering completed. {len(self.items)} of {total_count} in total.")
            torch.save(self.items, cache_path)

        self.length = len(self.items)
    
    def reorder_bbox(self, bboxs):

        for i, bbox in enumerate(bboxs):

            sort_bbox = bbox[np.argsort(bbox[:,0]), :]
            left, right = np.split(sort_bbox, 2, axis=0)

            left = left[np.argsort(left[:,1]), :]
            right = right[np.argsort(right[:,1]), :]

            bbox = np.stack((left[0], right[0], right[1], left[1]))
            bboxs[i] = bbox
        
        return bboxs
    

class JPosterDataset(ImageDataset):

    def __init__(self, cfgs, datype) -> None:
        super().__init__(cfgs, datype)

        # path
        image_root = ospj(cfgs.data_root, "images")
        cache_path = ospj(cfgs.data_root, f"{datype}_cache.pth")

        if os.path.exists(cache_path) and not cfgs.refresh:
            self.items = torch.load(cache_path)
        else:
            anno_root = ospj(cfgs.data_root, "annos", datype)
            anno_paths = glob.glob(ospj(anno_root, "*.json"))
            annos = []
            for anno_path in anno_paths:
                with open(anno_path, "r") as fp:
                    annos.append(json.load(fp))
            
            self.items = []
            total_count = 0
            for _ in range(self.repeat):
                for anno in tqdm(annos):
                    image_path = ospj(image_root, anno["imagePath"])
                    h, w = anno["imageHeight"], anno["imageWidth"]
                    sens = {}
                    shapes = anno["shapes"]
                    for shape in shapes:
                        idx = shape["group_id"]
                        if idx not in sens:
                            sens[idx] = [shape]
                        else:
                            sens[idx].append(shape)
                    
                    for sen in sens.values():
                        total_count += 1
                        label = ""
                        seg_bboxs = []
                        for ins in sen:
                            anno_type = ins["shape_type"]
                            if anno_type == "polygon":
                                seg_bbox = np.array(ins["points"], dtype=np.float32)
                            elif anno_type == "rectangle":
                                bbox = np.array(ins["points"], dtype=np.float32)
                                l, t, r, b = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
                                seg_bbox = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
                            else:
                                raise RuntimeError("illegal anno type")
                            seg_w = np.max(seg_bbox[:,0]) - np.min(seg_bbox[:,0])
                            seg_h = np.max(seg_bbox[:,1]) - np.min(seg_bbox[:,1])
                            if (seg_w/seg_h) > self.max_seg_aspect or (seg_h/seg_w) > self.max_seg_aspect: continue
                            if seg_bbox.shape[0] != 4: continue
                            label += ins["label"]
                            seg_bboxs.append(seg_bbox)

                        if len(label) == 0 or len(label) > self.seq_len: continue
                        seg_bboxs = np.stack(seg_bboxs, axis=0)
                        if np.any(seg_bboxs<0) or np.any(seg_bboxs[...,0]>w) or np.any(seg_bboxs[...,1]>h): continue

                        seg_bboxs = self.reorder_bbox(seg_bboxs)
                        seg_mask = np.array([1]*len(label) + [0]*(self.seq_len-len(label)))
                        mask_bbox, mask_area, seg_ratio = self.premeasure(h, w, seg_bboxs)
                        if seg_ratio < self.min_seg_ratio: continue

                        aug_type, aug_rbbox, seg_bboxs, mask_bbox, mask_rbbox = \
                            self.precalculate(h, w, seg_bboxs, mask_bbox, mask_area)

                        item = {
                            "image_path": image_path, 
                            "label": label,
                            "aug_type": aug_type,
                            "aug_rbbox": torch.tensor(aug_rbbox),
                            "seg_mask": torch.tensor(seg_mask),
                            "seg_bboxs": torch.tensor(seg_bboxs),
                            "mask_bbox": torch.tensor(mask_bbox),
                            "mask_rbbox": torch.tensor(mask_rbbox)
                        }

                        self.items.append(item)

            print(f"Data filtering completed. {len(self.items)} of {total_count} in total.")
            torch.save(self.items, cache_path)

        self.length = len(self.items)
    
    def reorder_bbox(self, bboxs):

        for i, bbox in enumerate(bboxs):

            sort_bbox = bbox[np.argsort(bbox[:,0]), :]
            left, right = np.split(sort_bbox, 2, axis=0)

            left = left[np.argsort(left[:,1]), :]
            right = right[np.argsort(right[:,1]), :]

            bbox = np.stack((left[0], right[0], right[1], left[1]))
            bboxs[i] = bbox
        
        return bboxs
            

def get_dataloader(cfgs):

    dataset_cfgs = OmegaConf.load(cfgs.dataset_cfg_path)
    print(f"Extracting data from {dataset_cfgs.target}")

    Dataset = eval(dataset_cfgs.target)
    dataset = Dataset(dataset_cfgs.params, datype = cfgs.datype)
    print(f"Dataset length: {len(dataset)}")

    return data.DataLoader(dataset=dataset, batch_size=cfgs.batch_size, shuffle=cfgs.shuffle,\
                            num_workers=cfgs.num_workers, drop_last=True, pin_memory=True)


if __name__ == "__main__":

    config_path = 'configs/test.yaml'
    cfgs = OmegaConf.load(config_path)

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.system("rm -rf sample/*")

    dataloader = get_dataloader(cfgs)

    for i, batch in enumerate(tqdm(dataloader)):

        if i>=100: continue

        save_dir = ospj("sample", f"s{i}")
        os.makedirs(save_dir, exist_ok=True)

        label = batch["label"][0]
        m_top, m_bottom, m_left, m_right = batch["r_bbox"][0]

        with open(ospj(save_dir, "label.txt"), "w") as fp:
            fp.write(batch["label"][0])

        save_image(batch["image"], ospj(save_dir, "image.png"), normalize=True)
        save_image(batch["mask"], ospj(save_dir, "mask.png"), normalize=True)
        save_image(batch["masked"], ospj(save_dir, "masked.png"), normalize=True)
        save_image(batch["image"][..., m_top:m_bottom, m_left:m_right], ospj(save_dir, "image_cropped.png"), normalize=True)

        if "seg_map" in batch:
            segs = batch["seg_map"].reshape(-1,1,512,512)
            seg_map = torch.zeros((1, 512, 512))
            for i, seg in enumerate(segs):
                seg_map += seg*(i+1)
            save_image(seg_map, ospj(save_dir, "seg_map.png"), normalize=True)

        if "c_ref" in batch:
            save_image(batch["c_ref"][0].unsqueeze(1), ospj(save_dir, "c_ref.png"), normalize=True)
        if "s_ref" in batch:
            save_image(batch["s_ref"][0], ospj(save_dir, "s_ref.png"), normalize=True)
        if "style" in batch:
            save_image(batch["style"], ospj(save_dir, "style.png"), normalize=True)
        if "ctrl" in batch:
            save_image(batch["ctrl"], ospj(save_dir, "ctrl.png"), normalize=True)
    

 