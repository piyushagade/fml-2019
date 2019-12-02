def image(_, data, resize=True):
    from PIL import Image, ImageOps
    desired_size = 40
    im = Image.fromarray(data)
    old_size = im.size  #
    ratio = 1
    if resize: 
        ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size)
    new_im = Image.new("I", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
    return _.np.pad(new_im, 10)