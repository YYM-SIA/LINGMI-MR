import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def draw_figure(fig):
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)


def show_tensor(a: torch.Tensor, fig_num=None, title=None, range=(None, None), ax=None):
    """Display a 2D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    a_np = a.squeeze().cpu().clone().detach().numpy()
    if a_np.ndim == 3:
        a_np = np.transpose(a_np, (1, 2, 0))

    if ax is None:
        fig = plt.figure(fig_num)
        plt.tight_layout()
        plt.cla()
        plt.imshow(a_np, vmin=range[0], vmax=range[1])
        plt.axis('off')
        plt.axis('equal')
        if title is not None:
            plt.title(title)
        draw_figure(fig)
    else:
        ax.cla()
        ax.imshow(a_np, vmin=range[0], vmax=range[1])
        ax.set_axis_off()
        ax.axis('equal')
        if title is not None:
            ax.set_title(title)
        draw_figure(plt.gcf())


def visual_rgb(a, name="image", permute=True, bgr=True, show=True):
    if a.shape[0] == 1:  # batch = 1
        x = a.squeeze().detach().cpu()
        if permute:
            x = x.permute([1, 2, 0])
        x = (x*255).int().numpy().astype(np.uint8)
        if bgr:
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        if(show): 
            cv2.imshow(name, x)
        return x
    else:
        out = []
        for i in range(a.shape[0]):
            # show each batch
            out.append(visual_rgb(a[i:i+1], name+"-%d" % (i), permute))
        return out


def visual_depth(a, name="image", cmap=cv2.COLORMAP_VIRIDIS, norm=True, min=None, max=None, show=True):
    if a.shape[0] == 1:  # batch = 1
        x = a.squeeze().detach().cpu()
        min = x.min() if min is None else min
        max = x.max() if max is None else max
        if norm:
            x[x < min] = min
            x[x > max] = max
            x = (x - min) / (max - min)
        x = (x*255).int().numpy().astype(np.uint8)
        if cmap is not None:
            x = cv2.applyColorMap(x, cmap)
        if(show):
            cv2.imshow(name, x)
        return x
    else:
        for i in range(a.shape[0]):
            # show each batch
            visual_depth(a[i:i+1], name+"-%d" % (i), cmap)
