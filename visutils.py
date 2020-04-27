from matplotlib import pyplot as plt

def show(imgs, title=None):

    plt.figure(figsize=(5 * len(imgs), 5))
    if title is not None:
        plt.suptitle(title + '\n', fontsize=24).set_y(1.05)

    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i + 1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0.02)

def part_grid(target_image, refernce_images, part_images):
    def proc(img):
        return (img * 255).permute(1, 2, 0).squeeze().cpu().numpy().astype('uint8')

    rows, cols = len(part_images) + 1, len(refernce_images) + 1
    fig = plt.figure(figsize=(cols*4, rows*4))
    sz = target_image.shape[-1]

    i = 1
    plt.subplot(rows, cols, i)
    plt.imshow(proc(target_image[0]))
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.title('Target', fontdict={'size': 26})

    for img in refernce_images:
        i += 1
        plt.subplot(rows, cols, i)
        plt.imshow(proc(img))
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.title('Reference', fontdict={'size': 26})

    for j, label in enumerate(part_images.keys()):
        i += 1
        plt.subplot(rows, cols, i)
        plt.imshow(proc(target_image[0]) * 0 + 255)
        plt.text(sz // 2, sz // 2, label.capitalize(), fontdict={'size': 30})
        plt.axis('off')
        plt.gca().set_axis_off()

        for img in part_images[label]:
            i += 1
            plt.subplot(rows, cols, i)
            plt.imshow(proc(img))
            plt.axis('off')
            plt.gca().set_axis_off()

        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.subplots_adjust(wspace=0, hspace=0)
    return fig


