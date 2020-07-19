from matplotlib.colors import ListedColormap

cmap_colours = [[0, 0, 0, 1],
                [0, 0, 0.1137, 1],
                [0, 0, 0.1373, 1],
                [0, 0, 0.1529, 1],
                [0, 0, 0.1686, 1],
                [0, 0, 0.1843, 1],
                [0, 0, 0.2000, 1],
                [0, 0, 0.2157, 1],
                [0, 0, 0.2275, 1],
                [0, 0, 0.2431, 1],
                [0, 0, 0.2627, 1],
                [0, 0, 0.2784, 1],
                [0, 0, 0.2902, 1],
                [0, 0, 0.3059, 1],
                [0, 0, 0.3255, 1],
                [0, 0, 0.3373, 1],
                [0, 0, 0.3529, 1],
                [0, 0, 0.3686, 1],
                [0.0078, 0, 0.3843, 1],
                [0.0196, 0, 0.4000, 1],
                [0.0275, 0, 0.4157, 1],
                [0.0431, 0, 0.4353, 1],
                [0.0627, 0, 0.4627, 1],
                [0.0745, 0, 0.4784, 1],
                [0.0824, 0, 0.4902, 1],
                [0.0941, 0, 0.5059, 1],
                [0.1059, 0, 0.5255, 1],
                [0.1176, 0, 0.5412, 1],
                [0.1255, 0, 0.5529, 1],
                [0.1373, 0, 0.5686, 1],
                [0.1490, 0, 0.5843, 1],
                [0.1608, 0, 0.6000, 1],
                [0.1686, 0, 0.6157, 1],
                [0.1804, 0, 0.6314, 1],
                [0.1922, 0, 0.6471, 1],
                [0.2039, 0, 0.6627, 1],
                [0.2157, 0, 0.6745, 1],
                [0.2235, 0, 0.6824, 1],
                [0.2353, 0, 0.6941, 1],
                [0.2471, 0, 0.7098, 1],
                [0.2588, 0, 0.7216, 1],
                [0.2667, 0, 0.7333, 1],
                [0.2745, 0, 0.7412, 1],
                [0.2941, 0, 0.7608, 1],
                [0.3137, 0, 0.7804, 1],
                [0.3216, 0, 0.7961, 1],
                [0.3294, 0, 0.8039, 1],
                [0.3412, 0, 0.8157, 1],
                [0.3569, 0, 0.8275, 1],
                [0.3647, 0, 0.8392, 1],
                [0.3725, 0, 0.8510, 1],
                [0.3843, 0, 0.8627, 1],
                [0.3961, 0, 0.8784, 1],
                [0.4078, 0, 0.8902, 1],
                [0.4196, 0, 0.9059, 1],
                [0.4275, 0, 0.9059, 1],
                [0.4392, 0, 0.9020, 1],
                [0.4510, 0, 0.8980, 1],
                [0.4627, 0, 0.8980, 1],
                [0.4706, 0, 0.8941, 1],
                [0.4824, 0, 0.8902, 1],
                [0.4941, 0, 0.8863, 1],
                [0.5059, 0, 0.8824, 1],
                [0.5137, 0, 0.8784, 1],
                [0.5373, 0, 0.8667, 1],
                [0.5490, 0, 0.8510, 1],
                [0.5569, 0, 0.8431, 1],
                [0.5686, 0, 0.8275, 1],
                [0.5804, 0, 0.8118, 1],
                [0.5922, 0, 0.8000, 1],
                [0.6039, 0, 0.7882, 1],
                [0.6118, 0, 0.7765, 1],
                [0.6157, 0, 0.7647, 1],
                [0.6235, 0, 0.7490, 1],
                [0.6275, 0, 0.7333, 1],
                [0.6314, 0, 0.7255, 1],
                [0.6353, 0, 0.7098, 1],
                [0.6392, 0, 0.6980, 1],
                [0.6471, 0, 0.6824, 1],
                [0.6510, 0, 0.6706, 1],
                [0.6588, 0, 0.6588, 1],
                [0.6627, 0, 0.6431, 1],
                [0.6667, 0, 0.6314, 1],
                [0.6706, 0, 0.6196, 1],
                [0.6745, 0, 0.6039, 1],
                [0.6784, 0, 0.5882, 1],
                [0.6902, 0, 0.5686, 1],
                [0.6980, 0, 0.5490, 1],
                [0.7020, 0, 0.5412, 1],
                [0.7059, 0, 0.5255, 1],
                [0.7098, 0, 0.5137, 1],
                [0.7176, 0, 0.5020, 1],
                [0.7216, 0, 0.4863, 1],
                [0.7255, 0, 0.4745, 1],
                [0.7294, 0, 0.4588, 1],
                [0.7373, 0, 0.4471, 1],
                [0.7412, 0, 0.4353, 1],
                [0.7451, 0, 0.4196, 1],
                [0.7490, 0, 0.4078, 1],
                [0.7569, 0, 0.3961, 1],
                [0.7608, 0, 0.3843, 1],
                [0.7647, 0, 0.3725, 1],
                [0.7686, 0, 0.3608, 1],
                [0.7765, 0, 0.3412, 1],
                [0.7804, 0, 0.3294, 1],
                [0.7882, 0.0039, 0.3176, 1],
                [0.7922, 0.0118, 0.3059, 1],
                [0.7961, 0.0235, 0.2863, 1],
                [0.8039, 0.0431, 0.2667, 1],
                [0.8118, 0.0510, 0.2510, 1],
                [0.8157, 0.0627, 0.2392, 1],
                [0.8196, 0.0706, 0.2275, 1],
                [0.8275, 0.0824, 0.2118, 1],
                [0.8314, 0.0902, 0.2000, 1],
                [0.8353, 0.0980, 0.1882, 1],
                [0.8392, 0.1098, 0.1765, 1],
                [0.8471, 0.1216, 0.1608, 1],
                [0.8510, 0.1294, 0.1490, 1],
                [0.8549, 0.1373, 0.1373, 1],
                [0.8588, 0.1490, 0.1216, 1],
                [0.8667, 0.1608, 0.1098, 1],
                [0.8706, 0.1686, 0.0980, 1],
                [0.8745, 0.1765, 0.0824, 1],
                [0.8784, 0.1882, 0.0706, 1],
                [0.8863, 0.2000, 0.0588, 1],
                [0.8902, 0.2078, 0.0431, 1],
                [0.8902, 0.2157, 0.0314, 1],
                [0.8980, 0.2235, 0.0196, 1],
                [0.9098, 0.2431, 0.0039, 1],
                [0.9176, 0.2588, 0, 1],
                [0.9216, 0.2667, 0, 1],
                [0.9216, 0.2745, 0, 1],
                [0.9294, 0.2824, 0, 1],
                [0.9373, 0.2980, 0, 1],
                [0.9412, 0.3059, 0, 1],
                [0.9412, 0.3137, 0, 1],
                [0.9490, 0.3216, 0, 1],
                [0.9569, 0.3373, 0, 1],
                [0.9569, 0.3451, 0, 1],
                [0.9608, 0.3529, 0, 1],
                [0.9686, 0.3608, 0, 1],
                [0.9765, 0.3765, 0, 1],
                [0.9765, 0.3843, 0, 1],
                [0.9843, 0.3922, 0, 1],
                [0.9922, 0.4000, 0, 1],
                [0.9922, 0.4118, 0, 1],
                [1.0000, 0.4235, 0, 1],
                [1.0000, 0.4275, 0, 1],
                [1.0000, 0.4353, 0, 1],
                [1.0000, 0.4431, 0, 1],
                [1.0000, 0.4510, 0, 1],
                [1.0000, 0.4588, 0, 1],
                [1.0000, 0.4667, 0, 1],
                [1.0000, 0.4745, 0, 1],
                [1.0000, 0.4824, 0, 1],
                [1.0000, 0.4863, 0, 1],
                [1.0000, 0.4941, 0, 1],
                [1.0000, 0.4980, 0, 1],
                [1.0000, 0.5059, 0, 1],
                [1.0000, 0.5137, 0, 1],
                [1.0000, 0.5176, 0, 1],
                [1.0000, 0.5255, 0, 1],
                [1.0000, 0.5333, 0, 1],
                [1.0000, 0.5373, 0, 1],
                [1.0000, 0.5451, 0, 1],
                [1.0000, 0.5490, 0, 1],
                [1.0000, 0.5569, 0, 1],
                [1.0000, 0.5647, 0, 1],
                [1.0000, 0.5725, 0, 1],
                [1.0000, 0.5765, 0, 1],
                [1.0000, 0.5843, 0, 1],
                [1.0000, 0.5922, 0, 1],
                [1.0000, 0.6000, 0, 1],
                [1.0000, 0.6118, 0, 1],
                [1.0000, 0.6157, 0, 1],
                [1.0000, 0.6235, 0, 1],
                [1.0000, 0.6275, 0, 1],
                [1.0000, 0.6353, 0, 1],
                [1.0000, 0.6431, 0, 1],
                [1.0000, 0.6510, 0, 1],
                [1.0000, 0.6549, 0, 1],
                [1.0000, 0.6627, 0, 1],
                [1.0000, 0.6667, 0, 1],
                [1.0000, 0.6745, 0, 1],
                [1.0000, 0.6824, 0, 1],
                [1.0000, 0.6863, 0, 1],
                [1.0000, 0.6941, 0, 1],
                [1.0000, 0.7020, 0, 1],
                [1.0000, 0.7098, 0, 1],
                [1.0000, 0.7137, 0, 1],
                [1.0000, 0.7176, 0, 1],
                [1.0000, 0.7216, 0, 1],
                [1.0000, 0.7333, 0, 1],
                [1.0000, 0.7490, 0, 1],
                [1.0000, 0.7529, 0, 1],
                [1.0000, 0.7569, 0, 1],
                [1.0000, 0.7608, 0, 1],
                [1.0000, 0.7725, 0, 1],
                [1.0000, 0.7804, 0, 1],
                [1.0000, 0.7843, 0, 1],
                [1.0000, 0.7882, 0, 1],
                [1.0000, 0.7961, 0, 1],
                [1.0000, 0.8000, 0, 1],
                [1.0000, 0.8118, 0, 1],
                [1.0000, 0.8157, 0, 1],
                [1.0000, 0.8196, 0, 1],
                [1.0000, 0.8275, 0, 1],
                [1.0000, 0.8392, 0, 1],
                [1.0000, 0.8431, 0, 1],
                [1.0000, 0.8471, 0, 1],
                [1.0000, 0.8549, 0, 1],
                [1.0000, 0.8588, 0, 1],
                [1.0000, 0.8667, 0, 1],
                [1.0000, 0.8745, 0, 1],
                [1.0000, 0.8824, 0, 1],
                [1.0000, 0.8902, 0, 1],
                [1.0000, 0.8980, 0, 1],
                [1.0000, 0.9059, 0, 1],
                [1.0000, 0.9137, 0, 1],
                [1.0000, 0.9216, 0, 1],
                [1.0000, 0.9255, 0, 1],
                [1.0000, 0.9294, 0, 1],
                [1.0000, 0.9373, 0, 1],
                [1.0000, 0.9451, 0.0196, 1],
                [1.0000, 0.9529, 0.0471, 1],
                [1.0000, 0.9569, 0.0745, 1],
                [1.0000, 0.9647, 0.1020, 1],
                [1.0000, 0.9725, 0.1294, 1],
                [1.0000, 0.9765, 0.1608, 1],
                [1.0000, 0.9843, 0.1882, 1],
                [1.0000, 0.9882, 0.2157, 1],
                [1.0000, 1.0000, 0.2431, 1],
                [1.0000, 1.0000, 0.2706, 1],
                [1.0000, 1.0000, 0.2980, 1],
                [1.0000, 1.0000, 0.3255, 1],
                [1.0000, 1.0000, 0.3529, 1],
                [1.0000, 1.0000, 0.4000, 1],
                [1.0000, 1.0000, 0.4392, 1],
                [1.0000, 1.0000, 0.4667, 1],
                [1.0000, 1.0000, 0.4980, 1],
                [1.0000, 1.0000, 0.5255, 1],
                [1.0000, 1.0000, 0.5529, 1],
                [1.0000, 1.0000, 0.5804, 1],
                [1.0000, 1.0000, 0.6078, 1],
                [1.0000, 1.0000, 0.6392, 1],
                [1.0000, 1.0000, 0.6627, 1],
                [1.0000, 1.0000, 0.6902, 1],
                [1.0000, 1.0000, 0.7216, 1],
                [1.0000, 1.0000, 0.7490, 1],
                [1.0000, 1.0000, 0.7765, 1],
                [1.0000, 1.0000, 0.8039, 1],
                [1.0000, 1.0000, 0.8314, 1],
                [1.0000, 1.0000, 0.8627, 1],
                [1.0000, 1.0000, 0.8863, 1],
                [1.0000, 1.0000, 0.9098, 1],
                [1.0000, 1.0000, 0.9412, 1]
                ]

default_cmap = ListedColormap(cmap_colours)
