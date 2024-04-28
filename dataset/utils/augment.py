'''Original code author: Li Hua'''
import numpy as np
import cv2 as cv
import random
from PIL import Image
from matplotlib import cm


# NUM_BIG_INT_
MAX_VALUE = 255
GAUSSIAN_SIGMA_THRES = 4

ALL_COLOR_MAPS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
				  'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                  'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                  'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                  'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                  'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                  'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
                  'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                  'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
                  'twilight', 'twilight_shifted', 'hsv',
                  'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                  'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b','tab20c',
                  'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                  'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                  'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                  'turbo', 'nipy_spectral', 'gist_ncar']


############################################################
##########          helper functions              ##########
############################################################

def get_random_color(n=1, max_value=MAX_VALUE):
    return np.round(np.random.rand(n,3)*max_value)

def get_random_sim_color(ref_color, max_value=MAX_VALUE):
    if random.random() < ref_color.mean()/max_value: # darker
        return ref_color*random.random()
    else: # lighter
        w = random.random()
        return w*ref_color + (1-w)*np.ones(3,)*max_value

'''
img: np array (size, size, 3), [0, 255]
color: [b, g, r]
apply color to gray scale image
'''
def apply_color(img, color, max_value=MAX_VALUE):
	color = color.reshape(-1,)
	color_img = np.stack((np.ones(img.shape[:2])*color[0], np.ones(img.shape[:2])*color[1], np.ones(img.shape[:2])*color[2]),axis=2)
	res = (max_value-(max_value-img)/max_value*(max_value-color_img))
	return res.astype(np.uint8)

def apply_color_img(img, color_img, max_value=MAX_VALUE):
	res = (max_value-(max_value-img)/max_value*(max_value-color_img))
	return res.astype(np.uint8)

'''
color: (b, g, r)
'''
def create_horizontal_gradient(h, w, start_color, end_color):
	gradient = np.zeros((h,w,3), np.uint8)
	gradient[:,:,:] = np.linspace(start_color, end_color, w, dtype=np.uint8)
	return gradient

'''
colors: (n_color, 3)
'''
def create_horizontal_gradient_multi(h, w, colors):
	gradient = np.zeros((h,w,3), np.uint8)
	n_color = colors.shape[0]
	ws = [w // (n_color-1)] * (n_color-1)
	ws[-1] += w - sum(ws)
	ws_cumsum = [0] + np.cumsum(ws).tolist()
	for i in range(n_color-1):
		gradient[:,ws_cumsum[i]:ws_cumsum[i+1],:] = np.linspace(colors[i], colors[i+1], ws[i], dtype=np.uint8)
	return gradient

'''
'''
def create_horizontal_colormap(h, w, colormap='rainbow', max_value=MAX_VALUE):
	colors = eval('cm.'+colormap)(np.arange(w)/w)
	gradient = np.zeros((h,w,3), np.uint8)
	gradient[:,:,:] = colors[:,:3]*max_value
	return gradient

'''
only for square
rot in degree
'''
def create_angled(h, func, PARAMS, rot=0):
	gradient = func(h*2, h*2, PARAMS)
	gradient = np.array(Image.fromarray(gradient).rotate(rot, resample=Image.Resampling.BILINEAR, expand=True))
	a, b = np.abs(np.sin(rot/180*np.pi))*h*2, np.abs(np.cos(rot/180*np.pi))*h*2
	start = int(np.ceil(a*b/(a+b)))
	gradient = gradient[start:-start, start:-start]
	gradient = cv.resize(gradient, (h, h))
	return gradient

'''
img: np array (size, size, 3), [0, 255]
color: [b, g, r]
assume img black front & white background, expand black border into color
'''
def create_bordered_img(img, color, width=1, gaussian=None):
	kernel = np.ones((int(2*width+1), int(2*width+1)), dtype=np.uint8)
	img_ = cv.erode(src=img, kernel=kernel)#, iterations=1)
	img_ = apply_color(img_, color)
	if gaussian and gaussian > 0:
		img_ = cv.GaussianBlur(img_, (gaussian*2+1, gaussian*2+1), 0)
	return img_

'''
colors: (n_color, 3) from inside to outside
widths: (n_color, ) from small to large
'''
def create_bordered_img_multi(img, colors, widths, gaussian=None):
	img_ = create_bordered_img(img, color=colors[-1], width=widths[-1])
	for color, width in zip(colors[::-1][1:], widths[::-1][1:]):
		img__ = create_bordered_img(img, color=color, width=width)
		ref = create_bordered_img(img, color=[0,0,0], width=width)
		img_ = overlay(img_, img__, ref)
	if gaussian and gaussian > 0:
		img_ = cv.GaussianBlur(img_, (gaussian*2+1, gaussian*2+1), 0)
	return img_

'''
img: np array (size, size, 3), [0, 255]
color: [b, g, r]
assume white background
'''
def create_shifted_img(img, color, distance=None, degree=45, bgcolor=[MAX_VALUE,MAX_VALUE,MAX_VALUE], gaussian=None):
	if not distance:
		distance = int(img.shape[0]*0.02)
	a, b = np.sin(degree/180*np.pi)*distance, np.cos(degree/180*np.pi)*distance
	img_ = Image.fromarray(img).rotate(0, translate=(int(np.round(a)),int(np.round(b))), fillcolor=tuple(bgcolor))
	img_ = apply_color(np.array(img_), color)
	if gaussian and gaussian > 0:
		img_ = cv.GaussianBlur(img_, (gaussian*2+1, gaussian*2+1), 0)
	return img_
    
'''
colors: (n_color, 3) from near to distant
distances: (n_color, ) from small to large
'''
def create_shifted_img_multi(img, colors, distances=None, degree=45, bgcolor=[MAX_VALUE,MAX_VALUE,MAX_VALUE], gaussian=None):
	if not distances:
		distances = [int(img.shape[0]*0.02), int(img.shape[0]*0.04), int(img.shape[0]*0.06)]
	img_ = create_shifted_img(img, color=colors[-1], distance=distances[-1], degree=degree, bgcolor=bgcolor)
	for color, distance in zip(colors[::-1][1:], distances[::-1][1:]):
		img__ = create_shifted_img(img, color=color, distance=distance)
		ref = create_shifted_img(img, color=[0,0,0], distance=distance)
		img_ = overlay(img_, img__, ref)
	if gaussian and gaussian > 0:
		img_ = cv.GaussianBlur(img_, (gaussian*2+1, gaussian*2+1), 0)
	return img_

############################################################
##########          end helper functions          ##########
############################################################

'''
ref_color: (b, g, r)
'''
def get_colored_img(img, p_black, p_single, p_single_grad, p_multi_grad, p_colormap, 
                        p_n=[0.8, 0.2], p_sim_color=0., ref_color=None, return_color=False):
    p = random.random()
    h, w, c = img.shape
    single = True
    if ref_color:
        if random.random() < p_sim_color: # apply similar color
            color = get_random_sim_color(ref_color)
            if random.random() < p_single/(p_single+p_single_grad): # single color
                pass
            else: # single color grad
                colors = np.stack((color, get_random_sim_color(color)), axis=0)
                gradient = create_angled(h, create_horizontal_gradient_multi, (colors), rot=random.random()*360)
                color = gradient

    elif p < p_black:
        if return_color:
            return np.ones_like(img)*MAX_VALUE, True # return blank bg
        else:
            return img, True # return original fontimg

    elif p < p_black + p_single:
        rndcolor = get_random_color(n=1)
        color = np.tile(rndcolor, (h,w,1)).astype(np.uint8)
    
    elif p < p_black + p_single + p_single_grad:
        rndcolor = get_random_color(n=1)
        colors = np.stack((rndcolor, get_random_sim_color(rndcolor)), axis=0)
        gradient = create_angled(h, create_horizontal_gradient_multi, (colors), rot=random.random()*360)
        color = gradient

    elif p < p_black + p_single + p_single_grad + p_multi_grad:
        colors = get_random_color(n=random.choices(range(2, len(p_n)+2),weights=p_n,k=1)[0])
        gradient = create_angled(h, create_horizontal_gradient_multi, (colors), rot=random.random()*360)
        color, single = gradient, False

    else:
        color_map = random.sample(ALL_COLOR_MAPS, 1)[0]
        gradient = create_angled(h, create_horizontal_colormap, (color_map), rot=random.random()*360)
        color, single = gradient, False

    if return_color:
        return color, single
    else:
        return apply_color_img(img, color), single

def get_colored_img_(img, p_black, p_single, p_single_grad, p_multi_grad, p_colormap, 
                        p_n=[0.8, 0.2], p_sim_color=0., ref_color=None, return_color=False):
    p = random.random()
    h, w, c = img.shape
    if ref_color:
        if random.random() < p_sim_color: # apply similar color
            color = get_random_sim_color(ref_color)
            if random.random() < p_single/(p_single+p_single_grad): # single color
                return apply_color(img, color), True
            else: # single color grad
                colors = np.stack((color, get_random_sim_color(color)), axis=0)
                gradient = create_angled(h, create_horizontal_gradient_multi, (colors), rot=random.random()*360)
                return apply_color_img(img, gradient), True
    if p < p_black:
        return img, True
    if p < p_black + p_single:
        color = get_random_color(n=1)
        return apply_color(img, color), True
    if p < p_black + p_single + p_single_grad:
        color = get_random_color(n=1)
        colors = np.stack((color, get_random_sim_color(color)), axis=0)
        gradient = create_angled(h, create_horizontal_gradient_multi, (colors), rot=random.random()*360)
        return apply_color_img(img, gradient), True
    if p < p_black + p_single + p_single_grad + p_multi_grad:
        colors = get_random_color(n=random.choices(range(2, len(p_n)+2),weights=p_n,k=1)[0])
        gradient = create_angled(h, create_horizontal_gradient_multi, (colors), rot=random.random()*360)
        return apply_color_img(img, gradient), False
    else:
        color_map = random.sample(ALL_COLOR_MAPS, 1)[0]
        gradient = create_angled(h, create_horizontal_colormap, (color_map), rot=random.random()*360)
        return apply_color_img(img, gradient), False

'''
single: if get_colored_img result is single-colored (or single grad)
'''
def get_bordered_img(img, single, p_border, min_width, max_width, min_gaussian, max_gaussian, 
                        color_single, color_multi, p_n=[0.8, 0.1, 0.1]):
    p = random.random()
    h, w, c = img.shape
    if p > p_border:
        return np.ones_like(img)*MAX_VALUE, np.ones_like(img)*MAX_VALUE
    # print('border')
    min_width, max_width = int(np.round(h*min_width)), int(np.round(h*max_width))
    min_gaussian, max_gaussian = int(np.round(h*min_gaussian)), int(np.round(h*max_gaussian))
    n_border = random.choices(range(1, len(p_n)+1), weights=p_n, k=1)[0]
    widths = sorted(np.abs(np.random.randn(n_border))/GAUSSIAN_SIGMA_THRES * (max_width-min_width) + min_width)
    gaussian = np.abs(np.random.randn(1))/GAUSSIAN_SIGMA_THRES * (max_gaussian-min_gaussian) + min_gaussian
    # widths = sorted(np.clip(np.abs(np.random.randn(n_border)) + min_width, a_min=min_width, a_max=max_width))
    # gaussian = np.clip(np.abs(np.random.randn(1)) + min_gaussian, a_min=min_gaussian, a_max=max_gaussian))
    border_ref = create_bordered_img_multi(img, np.array([[0,0,0]]), widths, int(np.round(gaussian)))
    if single:
        border, _ = get_colored_img(border_ref, **color_single)
    else:
        border, _ = get_colored_img(border_ref, **color_multi)
    return border, border_ref

'''
single: if get_colored_img result is single-colored (or single grad)
'''
def get_shifted_img(img, single, p_shift, min_distance, max_distance, min_gaussian, max_gaussian, 
                        color_single, color_multi, p_n=[0.8, 0.1, 0.1]):
    p = random.random()
    h, w, c = img.shape
    if p > p_shift:
        return np.ones_like(img)*MAX_VALUE, np.ones_like(img)*MAX_VALUE
    # print('shift')
    min_distance, max_distance = int(np.round(h*min_distance)), int(np.round(h*max_distance))
    min_gaussian, max_gaussian = int(np.round(h*min_gaussian)), int(np.round(h*max_gaussian))
    n_shift = random.choices(range(1, len(p_n)+1), weights=p_n, k=1)[0]
    distances = sorted(np.abs(np.random.randn(n_shift))/GAUSSIAN_SIGMA_THRES * (max_distance-min_distance) + min_distance)
    gaussian = np.abs(np.random.randn(1))/GAUSSIAN_SIGMA_THRES * (max_gaussian-min_gaussian) + min_gaussian
    shift_ref = create_shifted_img_multi(img, np.array([[0,0,0]]), distances, degree=random.random()*360, gaussian=int(np.round(gaussian)))
    if single:
        shift, _ = get_colored_img(shift_ref, **color_single)
    else:
        shift, _ = get_colored_img(shift_ref, **color_multi)
    return shift, shift_ref

'''
lay img2 on top of img1, filter out white color in img2 based on ref
img1, img2, ref: np array (size, size, 3), [0, 255]
'''
def overlay(img1, img2, ref, max_value=MAX_VALUE):
	# ref = np.array(Image.fromarray(ref).convert('L'))
	# print(ref.shape,ref[..., None].shape)
	img2 = Image.fromarray(np.concatenate([img2, max_value-ref[:,:,[0]]], axis=2).astype(np.uint8))
	res = Image.alpha_composite(Image.fromarray(img1).convert('RGBA'), img2)
	return np.array(res.convert('RGB'))
    

COLOR_TEXTURE_PARAMS = {
    'bg': {'p_black': 0.3, 'p_single': 0.3, 'p_single_grad': 0.1, 'p_multi_grad': 0.1, 'p_colormap': 0.2, 'p_n': [0.8, 0.2]},
    'color': {'p_black': 0.1, 'p_single': 0.2, 'p_single_grad': 0.2, 'p_multi_grad': 0.2, 'p_colormap': 0.3, 'p_n': [0.8, 0.2]},
    'border': {
        'p_border': 0.8,
        'min_width': 0.01, 'max_width': 0.08, 
        'min_gaussian': 0.0, 'max_gaussian': 0.1,
        'p_n': [0.8, 0.1, 0.1],
        'color_single': {'p_sim_color': 0.5, 'p_black': 0.0, 'p_single': 0.6, 'p_single_grad': 0.2, 'p_multi_grad': 0.1, 'p_colormap': 0.1, 'p_n': [0.8, 0.2]},
        'color_multi': {'p_black': 0.0, 'p_single': 0.8, 'p_single_grad': 0.2, 'p_multi_grad': 0.0, 'p_colormap': 0.0},
    },
    'shift': {
        'p_shift': 0.8,
        'min_distance': 0.01, 'max_distance': 0.08, 
        'min_gaussian': 0.0, 'max_gaussian': 0.1,
        'p_n': [0.8, 0.1, 0.1],
        'color_single': {'p_sim_color': 0.5, 'p_black': 0.3, 'p_single': 0.3, 'p_single_grad': 0.2, 'p_multi_grad': 0.1, 'p_colormap': 0.1, 'p_n': [0.8, 0.2]},
        'color_multi': {'p_black': 0.3, 'p_single': 0.5, 'p_single_grad': 0.2, 'p_multi_grad': 0.0, 'p_colormap': 0.0},
    }
}

'''
img: np array (size, size, 3), [0, 255], gray scale font img
seed is a big int
'''
def aug_method_color_texture(img, seed, color_texture_params=COLOR_TEXTURE_PARAMS, aug_bg=False):
    random.seed(seed)
    np.random.seed(seed)
    # seeds = np.random.rand(3,) # [0,1]
    if aug_bg:
        background, _ = get_colored_img(img, **color_texture_params['bg'], return_color=True)
    else:
        background = np.ones_like(img)*MAX_VALUE
    color, single = get_colored_img(img, **color_texture_params['color'])
    border, border_ref = get_bordered_img(img, single, **color_texture_params['border'])
    shift, shift_ref = get_shifted_img(img, single, **color_texture_params['shift']) # assume or not? shift based on shape of bordered img
    res = overlay(background, shift, shift_ref)
    res = overlay(res, border, border_ref) # overlay(shift, border, border_ref)
    res = overlay(res, color, img)
    return res


