def load(_, fname):
    with open(fname,'rb') as f:
        return _.pickle.load(f) if ".pkl" in fname else _.np.load(f, allow_pickle=True)
            
def write(_, fname,obj):
    with open(fname,'wb') as f:
        _.pickle.dump(obj,f)

def show_character(_, data, preprocess=True, label=None):
    if preprocess:
        _.plt.imshow(_.preprocess.image(_, _.np.matrix(data)))
    else:
        _.plt.imshow(_.np.matrix(data))

    if not label == None:
        _.plt.text(30, -5, label, horizontalalignment='center', verticalalignment='top')
    _.plt.show()

def show_characters(_, data):
    for i in range(100, 200):
        if i == 0:
            print(_.np.matrix(data[i]))
        _.plt.subplot(10, 10, i - 100 + 1)
        _.plt.axis('off')
        _.plt.imshow(_.preprocess.image(_, _.np.matrix(data[i])))
       
    # _.plt.text(-400.0, -700.0, 100, 200) 
    _.plt.get_current_fig_manager().window.showMaximized()
    _.plt.show()

def show_all_characters(_, data, start, end):
    for j in range(start, end):
        for i in range(j * 100 + 1, j * 100 + 101):
            _.plt.subplot(10, 10, i - 100 * j)
            _.plt.axis('off')
            _.plt.imshow(_.preprocess.image(_, _.np.matrix(data[i])))

        _.plt.get_current_fig_manager().window.showMaximized()
        _.plt.text(-400.0, -700.0, str(j * 100 + 1) + " - " + str(j * 100 + 101))
        _.plt.show()

def standardize(_, data_object, resize=True):
    standardized_data = []
    for i, item in enumerate(data_object):
        data_object[i] = _.np.matrix(data_object[i])
        data_object[i] = _.preprocess.image(_, data_object[i], resize)
        data_object[i] = _.np.array(data_object[i], dtype=_.np.int32)
        standardized_data.append(data_object[i])
    standardized_data = _.np.array(standardized_data)
    return standardized_data
